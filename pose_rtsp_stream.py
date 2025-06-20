import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
import subprocess
import queue
import logging
import os
import sys
import math

from modules.pose import Pose, track_poses
from modules.keypoints import extract_keypoints, group_keypoints


# ----------------- 配置参数 -----------------
# ONNX模型配置
ONNX_MODEL = 'human-pose-estimation.onnx'

# RTSP服务器配置
RTSP_SERVER = 'rtsp://192.168.137.35:8554/live'

# 摄像头配置
CAMERA_DEVICE = '/dev/video21'
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CAPTURE_FPS = 30

# 推流配置
STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080
STREAM_FPS = 30
STREAM_BITRATE = '4M'

# 姿态识别配置
POSE_WIDTH = 640
POSE_HEIGHT = 480
POSE_FPS = 15

# ONNX模型输入参数
INPUT_HEIGHT = 256
INPUT_WIDTH = 456
MEAN = np.array([128, 128, 128], dtype=np.float32)
SCALE = 1/256.0
STRIDE = 8
UPSAMPLE_RATIO = 4

# 队列设置
POSE_QUEUE_SIZE = 10
STREAM_QUEUE_SIZE = 30

# RK3588 NPU配置
USE_NPU = False

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)


# ----------------- 姿态估计相关函数 -----------------
    # 步骤：
    # 1. 解析模型输出的heatmaps和pafs
    # 2. 使用双三次插值上采样
    # 3. 根据stride和upsample_ratio调整尺寸
    
def preprocess(img):
    """预处理输入图像"""
    h, w, _ = img.shape
    scale = min(INPUT_HEIGHT / h, INPUT_WIDTH / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))
    img_norm = (img_resized.astype(np.float32) - MEAN) * SCALE
    
    pad_top = (INPUT_HEIGHT - nh) // 2
    pad_bottom = INPUT_HEIGHT - nh - pad_top
    pad_left = (INPUT_WIDTH - nw) // 2
    pad_right = INPUT_WIDTH - nw - pad_left
    
    img_pad = cv2.copyMakeBorder(
        img_norm, pad_top, pad_bottom, pad_left, pad_right, 
        cv2.BORDER_CONSTANT, value=0
    )
    img_input = np.transpose(img_pad, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
    
    return img_input, scale, [pad_top, pad_left, pad_bottom, pad_right]

def postprocess(outputs, scale, pad):
    """后处理模型输出"""
    heatmaps = np.transpose(outputs[-2].squeeze(), (1, 2, 0))
    pafs = np.transpose(outputs[-1].squeeze(), (1, 2, 0))
    
    heatmaps = cv2.resize(
        heatmaps, (0, 0), 
        fx=UPSAMPLE_RATIO, fy=UPSAMPLE_RATIO, 
        interpolation=cv2.INTER_CUBIC
    )
    pafs = cv2.resize(
        pafs, (0, 0), 
        fx=UPSAMPLE_RATIO, fy=UPSAMPLE_RATIO, 
        interpolation=cv2.INTER_CUBIC
    )
    
    return heatmaps, pafs

    # 完整流程：
    # 1. 预处理
    # 2. ONNX推理
    # 3. 后处理
    # 4. 关键点提取（extract_keypoints）
    # 5. 关键点分组（group_keypoints）
    # 6. 坐标转换（回到原始图像坐标系）
def detect_poses(img, ort_session):
    """检测姿态 - 使用与demo.py完全相同的算法"""
    img_input, scale, pad = preprocess(img)
    ort_inputs = {ort_session.get_inputs()[0].name: img_input}
    outputs = ort_session.run(None, ort_inputs)
    heatmaps, pafs = postprocess(outputs, scale, pad)
    
    # 使用完整的关键点提取和分组算法
    total_keypoints_num = 0
    all_keypoints_by_type = []
    num_keypoints = Pose.num_kpts
    
    for kpt_idx in range(num_keypoints):
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
    
    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    
    # 坐标转换
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * STRIDE / UPSAMPLE_RATIO - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * STRIDE / UPSAMPLE_RATIO - pad[0]) / scale
    
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)
    
    return current_poses

def draw_poses(img, poses):
    """绘制姿态 - 与demo.py完全相同的绘制方式"""
    orig_img = img.copy()
    
    for pose in poses:
        pose.draw(img)
    
    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    
    for pose in poses:
        pose: Pose
        cv2.rectangle(
            img,
            (pose.bbox[0], pose.bbox[1]),
            (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]),
            (0, 0, 255) if pose.is_fallen_down() else (0, 255, 0),
            5 if pose.is_fallen_down() else 2,
        )
        cv2.putText(
            img,
            f'id: {pose.id}, fallen down!' if pose.is_fallen_down() else f'id: {pose.id}',
            (pose.bbox[0], pose.bbox[1] - 16),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 0, 255) if pose.is_fallen_down() else (0, 255, 0),
            1,
        )
    
    return img

# ----------------- 摄像头捕获线程 -----------------
        # 工作流程：
        # 1. 打开摄像头并设置参数
        # 2. 循环捕获帧：
        #   - 添加到推流队列（stream_queue）
        #   - 按POSE_FPS抽取帧添加到姿态队列（pose_queue）
        # 3. 设置缓冲区大小为1减少延迟
class CameraCapture(threading.Thread):
    def __init__(self, device, width, height, fps):
        super().__init__(name="CameraCapture")
        self.device = device  	# 摄像头设备ID
        self.width = width
        self.height = height
        self.fps = fps		# 目标帧率
        self.cap = None
        self.running = False
        #降采样帧（用于姿态识别，降低计算负载），队列大小由POSE_QUEUE_SIZE控制）
        self.pose_queue = queue.Queue(maxsize=POSE_QUEUE_SIZE)
        #全帧率原始帧（用于实时推流），队列大小由STREAM_QUEUE_SIZE控制）
        self.stream_queue = queue.Queue(maxsize=STREAM_QUEUE_SIZE)
        
    def run(self):
        # 打开摄像头
        self.cap = cv2.VideoCapture(self.device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)# 关键！减少延迟
        
        if not self.cap.isOpened():
            logging.error(f"无法打开摄像头: {self.device}")
            return
            
        logging.info(f"摄像头已打开: {self.device} {self.width}x{self.height}@{self.fps}fps")
        
        self.running = True
        frame_count = 0
        pose_frame_interval = CAPTURE_FPS // POSE_FPS
        
        # 主循环
        while self.running:
        # 读取一帧
            ret, frame = self.cap.read() # 捕获一帧
            if not ret:
                logging.error("读取帧失败")
                continue
            
            #  推流队列处理（全帧率）：每帧都存入，队列满时丢弃最旧帧（保证实时性）
            try:
                if self.stream_queue.full():
                    self.stream_queue.get_nowait()		# 丢弃最旧帧
                self.stream_queue.put_nowait(frame.copy())	# 存入新帧
            except queue.Full:
                pass
            
            # 姿态识别队列处理（降采样）：按固定间隔（pose_frame_interval）抽取帧
            if frame_count % pose_frame_interval == 0:
                try:
                    pose_frame = cv2.resize(frame, (POSE_WIDTH, POSE_HEIGHT))	# 降采样
                    if self.pose_queue.full():
                        self.pose_queue.get_nowait()				# 丢弃最旧帧
                    self.pose_queue.put_nowait(pose_frame)			# 存入新帧
                except queue.Full:
                    pass
            
            frame_count += 1							# 帧计数器递增
            
    def stop(self):
        self.running = False		# 终止循环
        if self.cap:
            self.cap.release()		# 释放摄像头
        logging.info("摄像头已关闭")

# ----------------- RTSP推流线程 -----------------
        # 工作流程：
        # 1. 构造ffmpeg命令（优先使用硬件编码）
        # 2. 启动ffmpeg进程
        #硬件加速优先：优先使用RK3588硬件编码器(h264_rkmpp)，失败时自动回退到软件编码(libx264)
        # 3. 从流队列获取帧：
        #   - 调整尺寸（如果需要）
        #   - 写入ffmpeg标准输入
        # 4. 处理异常（自动回退到软件编码）
class RTSPStreamer(threading.Thread):
    def __init__(self, rtsp_url, width, height, fps, bitrate, input_queue):
        super().__init__(name="RTSPStreamer")
        self.rtsp_url = rtsp_url		# RTSP服务器地址
        self.width = width			# 推流宽度
        self.height = height			# 推流高度
        self.fps = fps				# 推流帧率
        self.bitrate = bitrate			# 视频码率（如"2000k"）
        self.input_queue = input_queue		# 输入队列（来自CameraCapture）
        self.running = False			# 运行标志
        self.ffmpeg_process = None		# ffmpeg进程句柄
        
    def run(self):
        command = [
            'ffmpeg',
            '-f', 'rawvideo',			# 输入格式：原始视频
            '-pix_fmt', 'bgr24',		# OpenCV默认的BGR24格式
            '-s', f'{self.width}x{self.height}',# 帧尺寸
            '-r', str(self.fps),		# 帧率
            '-i', '-',				# 从标准输入读取
            '-c:v', 'h264_rkmpp',		# 优先使用RK3588硬件编码
            '-b:v', self.bitrate,		# 视频码率
            '-preset', 'ultrafast',		# 最快编码预设
            '-tune', 'zerolatency',		# 零延迟调优
            '-g', str(self.fps),		# GOP大小（关键帧间隔=1秒）
            '-f', 'rtsp',			# 输出格式：RTSP
            '-rtsp_transport', 'tcp',		# 使用TCP传输（更可靠）
            self.rtsp_url			# RTSP服务器URL
        ]
        
        #ffmpeg进程启动：整个编码工作由ffmpeg在独立的进程中处理
        try:
        # 尝试硬件编码
            self.ffmpeg_process = subprocess.Popen(
                command, 
                stdin=subprocess.PIPE,			# 标准输入管道：建了一个通向ffmpeg进程的管道
                stderr=subprocess.PIPE			# 捕获错误输出
            )
        except:
        # 硬件编码失败时回退到软件编码
            logging.warning("RK3588硬件编码器不可用，使用软件编码")
            command[10] = 'libx264'			# 替换编码器为软件编码
            self.ffmpeg_process = subprocess.Popen(
                command, 
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        logging.info(f"RTSP推流已启动: {self.rtsp_url}")
        self.running = True
        
        #帧处理循环
        while self.running:
            try:
            						# 从队列获取帧（最多等待1秒）
                frame = self.input_queue.get(timeout=1.0)
                if frame is None:
                    break
                    					# 尺寸检查与调整
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                    					# 写入ffmpeg标准输入
                self.ffmpeg_process.stdin.write(frame.tobytes()) #此时ffmpeg接收到原始帧数据并开始编码
                self.ffmpeg_process.stdin.flush()	# 立即发送
                
            except queue.Empty:				# 队列空时继续尝试
                continue
            except Exception as e:			# 其他错误处理
                logging.error(f"推流错误: {str(e)}")
                break
                
    def stop(self):
        self.running = False
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()		# 关闭输入管道
            self.ffmpeg_process.terminate()		# 发送终止信号
            self.ffmpeg_process.wait()			# 等待进程退出
        logging.info("RTSP推流已停止")

# ----------------- 姿态估计线程 -----------------
	#人体姿态估计：使用ONNX模型检测人体关键点
	#姿态跟踪：跨帧关联同一人体的姿态
	#结果可视化：绘制关键点、骨骼连接和性能信息
	#NPU加速支持：可选用RK3588的NPU进行加速推理
class PoseEstimator(threading.Thread):
    def __init__(self, model_path, input_queue):
        super().__init__(name="PoseEstimator")
        self.model_path = model_path			# ONNX模型路径
        self.input_queue = input_queue			# 输入队列（来自CameraCapture）
        self.running = False
        self.ort_session = None				# ONNX Runtime推理会话
        self.previous_poses = []  			# 用于姿态跟踪，存储上一帧的姿态信息，用于跨帧跟踪
        
    def run(self):
        # 加载ONNX模型
        ## 根据是否使用NPU选择执行提供者
        providers = ['CPUExecutionProvider']
        if USE_NPU:
            providers = ['RknpuExecutionProvider', 'CPUExecutionProvider']	# NPU优先
            									# 创建ONNX Runtime推理会话
        self.ort_session = ort.InferenceSession(
            self.model_path, 
            providers=providers
        )
        logging.info(f"ONNX模型已加载: {self.model_path}")
        
        self.running = True
        
        while self.running:
            try:
                frame = self.input_queue.get(timeout=1.0)			# 从队列获取帧（超时1秒）
                if frame is None:
                    break
                    
                # 姿态估计
                t1 = time.time()
                				#调用detect_poses函数进行关键点检测
                current_poses = detect_poses(frame, self.ort_session)
                t2 = time.time()
                inference_time = int((t2 - t1) * 1000)				# 计算推理时间(ms)
                
                # 姿态跟踪
                if len(self.previous_poses) > 0:
                				#track_poses函数：跨帧关联同一人体的姿态
                    track_poses(self.previous_poses, current_poses, threshold=3, smooth=True)
                self.previous_poses = current_poses				#存储当前帧姿态用于下一帧跟踪
                
                # 绘制结果：在图像上绘制检测结果
                result_img = draw_poses(frame, current_poses, track=True, smooth=True)
                
                # 统计有效关键点数量
                total_keypoints = 0
                for pose in current_poses:
                    valid_keypoints = sum(1 for kp in pose.keypoints if kp[0] > 0 and kp[1] > 0)
                    total_keypoints += valid_keypoints
                
                # 添加性能信息
                cv2.putText(
                    result_img, 
                    f'Inference: {inference_time}ms | FPS: {POSE_FPS} | Poses: {len(current_poses)} | KPs: {total_keypoints}', 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # 显示结果
                cv2.imshow('Pose Estimation', result_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"姿态估计错误: {str(e)}")
                import traceback
                traceback.print_exc()
                
    def stop(self):
        self.running = False
        cv2.destroyAllWindows()
        logging.info("姿态估计已停止")

# ----------------- 主函数 -----------------
def main():
    try:
        # 检查模型文件
        if not os.path.exists(ONNX_MODEL):
            logging.error(f"找不到ONNX模型文件: {ONNX_MODEL}")
            return
            
        # 创建摄像头捕获线程，摄像头捕获线程专注于高效获取原始视频帧
        camera = CameraCapture(CAMERA_DEVICE, FRAME_WIDTH, FRAME_HEIGHT, CAPTURE_FPS)
        
        # 创建RTSP推流线程，RTSP推流线程专注于视频编码和网络传输
        streamer = RTSPStreamer(
            RTSP_SERVER, 
            STREAM_WIDTH, 
            STREAM_HEIGHT, 
            STREAM_FPS, 
            STREAM_BITRATE,
            camera.stream_queue
        )
        
        # 创建姿态估计线程，姿态估计线程专注于计算密集型的人体姿态识别
        pose_estimator = PoseEstimator(ONNX_MODEL, camera.pose_queue)
        
        # 启动所有线程
        camera.start()
        time.sleep(1)
        
        streamer.start()
        pose_estimator.start()
        
        logging.info("系统已启动，按Ctrl+C退出")
        
        # 主线程等待
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("收到退出信号")
            
        # 停止所有线程
        camera.stop()
        streamer.stop()
        pose_estimator.stop()
        
        # 等待线程结束
        camera.join()
        streamer.join()
        pose_estimator.join()
        
        logging.info("程序已退出")
        
    except Exception as e:
        logging.error(f"主程序错误: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 

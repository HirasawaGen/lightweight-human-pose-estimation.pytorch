from typing import Sequence
import cv2
import os
import time
import numpy as np
import torch
import requests

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
from loguru import logger


# VIDEO_PATH = os.environ['VIDEO_PATH']
# EQUIPMENT_ID = os.environ['EQUIPMENT_ID']
# API_URL_FORMAT = os.environ['API_URL_FORMAT']
# MODEL_PATH = os.environ['MODEL_PATH']
# ALERT_THRESHOLD = float(os.environ['ALERT_THRESHOLD'])
# EMAIL_SEND_INTERVAL = float(os.environ['EMAIL_SEND_INTERVAL'])

VIDEO_PATH = './data/59789586-1-192.mp4'  # 本地测试
EQUIPMENT_ID = 'YSU123'
API_URL_FORMAT = 'https://localhost:8080/send-email?id={}'
API_URL = API_URL_FORMAT.format(EQUIPMENT_ID)
MODEL_PATH = './checkpoints/checkpoint_iter_370000.pth'
ALERT_THRESHOLD = 3.0
EMAIL_SEND_INTERVAL = 20.0

to_cuda = True
display = True

class VideoReader:
    def __init__(self, device_path: str, fps: int=10):
        self._device_path = device_path
        self._fps = fps
        self._managed = False
        self._cap: cv2.VideoCapture
    
    def __enter__(self):
        self._cap = cv2.VideoCapture(self._device_path)
        self._managed = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._managed:
            return
        self._cap.release()
        self._managed = False
        if exc_type is KeyboardInterrupt:
            logger.info('KeyboardInterrupt')
        logger.success('VideoReader released successfully')

            
    def __iter__(self) -> 'VideoReader':
        if not self._managed:
            raise RuntimeError('VideoReader is not managed')
        return self
    
    def __next__(self) -> cv2.typing.MatLike:
        if not self._managed:
            raise RuntimeError('VideoReader is not managed')
        if not self._cap.isOpened():
            raise StopIteration
        ret, frame = self._cap.read()
        if not ret:
            raise StopIteration
        if self._fps > 0:
            print(f'frame rate: {self._fps}')
            time.sleep(1 / self._fps)
        return frame


reader: VideoReader


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if to_cuda:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def main():
    fallen_time_map = {}
    logger.info('Loading model...')
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    load_state(net, checkpoint)
    logger.success('Model loaded successfully')
    net = net.eval()
    if to_cuda:
        net = net.cuda()
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses: Sequence[Pose] = []
    delay = 0
    for img in reader:
        orig_img = None if not display else img.copy()
        heatmaps, pafs, scale, pad = infer_fast(
            net,
            img,
            512,
            stride,
            upsample_ratio,
            False
        )
        
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses: Sequence[Pose] = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        track_poses(previous_poses, current_poses, smooth=True)
        previous_poses = current_poses
        # print(fallen_time_map)
        detected_ids = set()
        for pose in current_poses:
            pose_id = str(pose.id)
            detected_ids.add(pose_id)
            # TODO: 如果逻辑没问题的话，就让ai改写成策略模式  补：ai猪脑子写不明白策略模式
            if pose.is_fallen_down():
                if pose_id not in fallen_time_map:
                    fallen_time_map[pose_id] = {
                        'fallen_time': time.time(),
                        'email_sent': False
                    }
                else:
                    fallen_time = fallen_time_map[pose_id]['fallen_time']
                    email_sent = fallen_time_map[pose_id]['email_sent']
                    fallen_during = time.time() - fallen_time
                    if fallen_during < ALERT_THRESHOLD:
                        continue
                    if fallen_during > ALERT_THRESHOLD:
                        if not email_sent:
                            # TODO: send email use requests
                            fallen_time_map[pose_id]['email_sent'] = True
                            fallen_time_map[pose_id]['last_alert_time'] = time.time()
                        else:
                            if time.time() - fallen_time_map[pose_id]['last_alert_time'] > EMAIL_SEND_INTERVAL:
                                # TODO: send email use requests
                                fallen_time_map[pose_id]['last_alert_time'] = time.time()
            else:
                if pose_id in fallen_time_map:
                    fallen_time_map.pop(pose_id)
        diff = fallen_time_map.keys() - detected_ids
        for pose_id in diff:
            fallen_time_map.pop(pose_id)
        
                    
        fallen_down_count = len(fallen_time_map)
        if fallen_down_count > 0:
            logger.warning(f'dectected {fallen_down_count} people fallen down.')
        else:
            logger.info('no fallen down detected.')
            
        if display:
            for pose in current_poses:
                pose.draw(img)
            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            for pose in current_poses:
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
            cv2.imshow('Lightweight Human Pose Estimation', img)
            key = cv2.waitKey(delay)
            if key == 27:  # esc
                return
            elif key == 32:  # space
                delay = 0 if delay == 1 else 1

        
if __name__ == '__main__':
    reader = VideoReader(VIDEO_PATH, -1)
    with reader:
        main()
        
    
    
        
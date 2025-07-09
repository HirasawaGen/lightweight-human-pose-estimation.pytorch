该项目fork自[lightweight-human-pose-estimation.pytorch](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)，并进行了一些修改。

主要修改了`.\modules\pose.py`这一文件，为`Pose`类添加了判断是否为跌倒状态、是否为合法姿态的函数。

新建了`.\main.py`文件，用来检测摄像头或者视频中的人体姿态，如果为跌倒状态，则向远程服务器发送请求，并由远程服务器使用SMTP协议为用户发送警告邮件。

以下视频为样例输入1：

<video controls>
    <source src="./data/input.mp4" type="video/mp4">
</video>

经过模型预测到人体姿态并绘制在视频上，样例输出1如下：

<video controls>
    <source src="./result/output.mp4" type="video/mp4">
</video>
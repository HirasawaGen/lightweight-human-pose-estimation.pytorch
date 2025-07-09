该项目fork自[lightweight-human-pose-estimation.pytorch](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)，并进行了一些修改。

展示视频：`.\video.mp4`

主要修改了`.\modules\pose.py`这一文件，为`Pose`类添加了判断是否为跌倒状态、是否为合法姿态的函数。

新建了`.\main.py`文件，用来检测摄像头或者视频中的人体姿态，如果为跌倒状态，则向远程服务器发送请求，并由远程服务器使用SMTP协议为用户发送警告邮件。

样例输入视频文件为：`.\data\input.mp4`

样例输出视频文件为：`.\result\output.mp4`

技术文档：`.\技术文档.pdf`

[点击查看原项目README.md](.\orig_README.md)
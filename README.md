# opencv

## 如何开始
  1. 确保本地有opencv库  
    `pip3 install opencv-python opencv-contrib-python`
  2. 运行`join.py`文件 拍摄美照, 需要输入id（可以自定义自己照片的id，可以多人）
  3. 运行`train.py`， 之后程序会自动生成人脸数据训练集trainer.yml
  4. 运行`run.py` 就可以对已经录入的人脸进行识别

import cv2
import time
stream_url = "http://192.168.42.2:40921"

# 创建视频捕获对象
#cap = cv2.VideoCapture(stream_url)
cap = cv2.VideoCapture(stream_url, cv2.CAP_GSTREAMER)
# while True:
#     # 读取视频帧
#     ret, frame = cap.read()

#     # 如果读取成功，显示帧
#     if ret:
#         print(frame)
#     else:
#         print("无法读取帧")
#         break

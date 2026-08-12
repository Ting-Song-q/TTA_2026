#!/usr/bin/python3
# coding=UTF-8

import cv2
from robomaster import robot
from robomaster import camera
import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge , CvBridgeError
import time
from sensor_msgs.msg import CameraInfo
import yaml
import time

if __name__ == '__main__':
    ep_robot = robot.Robot()

    ep_robot.initialize(conn_type='rndis')
    ep_camera = ep_robot.camera
    print(ep_camera.video_stream_addr)
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    
    count = 0
    while not rospy.is_shutdown():    # Ctrl C正常退出，如果异常退出会报错device busy！
        time.sleep(1)
        count+=1
        img = ep_camera.read_cv2_image(timeout=3,strategy='newest')
        cv2.imwrite("{"+str(count)+"}.jpg", img)
        print("write an image")

    ep_camera.stop_video_stream()
    # ep_robot.close()
    print("quit successfully!")

  
    # for i in range(0, 200):
    #     img = ep_camera.read_cv2_image()
    #     cv2.imshow("Robot", img)
    #     cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # ep_camera.stop_video_stream()
    # ep_robot.close()

#!/usr/bin/python3
# coding=UTF-8

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import numpy as np
import cv2

class take_picture:
    def __init__(self):
        rospy.init_node('picture_getter', anonymous=True)
        self.image = None
        self.cv_bridge = CvBridge()
        self.camera_info_sub  = rospy.Subscriber("ep_cam/image_raw", Image,
                                                 self.camera_info_cb, queue_size=1)
        self.take_picture()
    
    def camera_info_cb(self, msg):
        # 拷贝一份，避免后续处理时踩内存
        self.image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8").copy()
       
    def take_picture(self):
        # 相机参数只算一次即可
        dist_coeffs = np.array([-0.0444954569665007, -0.00201770323876866,
                                0, 0, -0.0159863040899372])
        intrinsic = np.array([[624.302683276108, 0, 632.376721286788],
                              [0, 623.915644264347, 371.420934824104],
                              [0, 0, 1]])

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if input().strip() == 'k':
                if self.image is None:
                    rospy.logwarn("还没有收到图像，请稍后再按 k")
                    continue

                time = rospy.Time.now()
                # 用新的变量保存畸变结果
                corrected_image = cv2.undistort(self.image,
                                                intrinsic,
                                                dist_coeffs)
                cv2.imwrite(
                    f'/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/{time}.jpg',
                    corrected_image)
            rate.sleep()

if __name__ == '__main__':
    picture_getter = take_picture()
    rospy.spin()
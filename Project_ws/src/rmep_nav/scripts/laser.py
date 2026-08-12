#!/usr/bin/python3
# coding=UTF-8

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from tf.transformations import euler_from_quaternion

from collections import deque

class LidarUndistortion:
    def __init__(self):
        # ³õÊ¼»¯ ROS ½Úµã
        rospy.init_node('lidar_undistortion_node')

        # »º´æÊý¾Ý¶ÓÁÐ
        self.imu_data = deque()
        self.odom_data = deque()

        # ¶©ÔÄ IMU ºÍ Àï³Ì¼ÆÊý¾Ý
        rospy.Subscriber('/imu_data', Imu, self.imu_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # ´´½¨Ò»¸ö·¢²¼Ð£ÕýºóµãÔÆµÄ·¢²¼Õß
        self.corrected_pointcloud_pub = rospy.Publisher('/corrected_pointcloud', PointCloud2, queue_size=1)

    def imu_callback(self, imu_msg):
        # ½«½ÓÊÕµ½µÄ IMU Êý¾Ý±£´æµ½¶ÓÁÐÖÐ
        self.imu_data.append(imu_msg)
    
    def odom_callback(self, odom_msg):
        # ½«½ÓÊÕµ½µÄÀï³Ì¼ÆÊý¾Ý±£´æµ½¶ÓÁÐÖÐ
        self.odom_data.append(odom_msg)

    def scan_callback(self, scan_msg):
        # ¼¤¹âÉ¨ÃèÊý¾ÝµÄ»Øµ÷º¯Êý£¬´¦ÀíÈ¥»û±ä
        if not self.imu_data or not self.odom_data:
            rospy.logwarn("Waiting for IMU and Odom data...")
            return

        # »ñÈ¡É¨ÃèÊ±¼ä´Á
        scan_time = scan_msg.header.stamp

        # ÐÞ¼ô IMU ºÍ Àï³Ì¼ÆÊý¾Ý¶ÓÁÐ£¬È·±£ÓëÉ¨ÃèÊ±¼ä´ÁÒ»ÖÂ
        self.trim_data_queue(scan_time)

        # »º´æ½Ç¶È
        angle_cache = self.create_angle_cache(scan_msg)

        # È¥»û±äÐ£Õý
        corrected_points = self.correct_laser_scan(scan_msg, angle_cache)

        # ·¢²¼Ð£ÕýºóµÄµãÔÆÊý¾Ý
        self.publish_corrected_pointcloud(corrected_points, scan_msg.header)

    def trim_data_queue(self, scan_time):
        # ÐÞ¼ô IMU ºÍ Àï³Ì¼ÆÊý¾Ý¶ÓÁÐ
        while self.imu_data and self.imu_data[0].header.stamp < scan_time:
            self.imu_data.popleft()
        while self.odom_data and self.odom_data[0].header.stamp < scan_time:
            self.odom_data.popleft()

    def create_angle_cache(self, scan_msg):
        # »º´æ¼¤¹âÉ¨ÃèµÄ½Ç¶ÈÐÅÏ¢
        num_points = len(scan_msg.ranges)
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        return [angle_min + i * angle_increment for i in range(num_points)]

    def correct_laser_scan(self, scan_msg, angle_cache):
        # Ê¹ÓÃ IMU ºÍ Àï³Ì¼ÆÊý¾Ý½øÐÐÈ¥»û±äÐ£Õý
        corrected_points = []

        for i, range_val in enumerate(scan_msg.ranges):
            if np.isinf(range_val) or np.isnan(range_val):
                continue

            angle = angle_cache[i]
            x = range_val * np.cos(angle)
            y = range_val * np.sin(angle)

            # TODO: ¸ù¾Ý IMU ºÍÀï³Ì¼ÆÊý¾Ý½øÐÐÐ£Õý£¬µ÷Õû (x, y) ×ø±ê

            corrected_points.append((x, y, 0))  # ¼ÙÉèÆ½ÃæÄÚµÄz×ø±êÎª0

        return corrected_points

    def publish_corrected_pointcloud(self, points, header):
        # ´´½¨²¢·¢²¼Ð£ÕýºóµÄµãÔÆ
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        pointcloud = point_cloud2.create_cloud(header, fields, points)
        self.corrected_pointcloud_pub.publish(pointcloud)

if __name__ == "__main__":
    # ´´½¨ LidarUndistortion ¶ÔÏó
    node = LidarUndistortion()
    rospy.spin()

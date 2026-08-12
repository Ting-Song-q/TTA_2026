#!/usr/bin/python3
# coding=UTF-8

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math
import numpy as np

class initPosition:
    def __init__(self):
        rospy.init_node('init_position', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_subscriber = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.laser_data = None
        self.rate = rospy.Rate(10)
        self.move_to_initial_position(0.75, 0.75)
    
    def laser_callback(self, laser_data):
        self.laser_data = laser_data
        

    def move_to_initial_position(self, x, y):
        self.align_wall()
        while not rospy.is_shutdown():
            if self.laser_data:
                left_distance = min(self.laser_data.ranges[764:956])
                first_30de = self.laser_data.ranges[:95]
                last_30de = self.laser_data.ranges[-95:]
                combined = np.concatenate((first_30de, last_30de))
                back_distance = np.min(combined)
                print(back_distance)
                if abs(left_distance - x) > 0.01:
                    if left_distance  > x:
                        vel_msg = Twist()
                        vel_msg.linear.y = 0.1
                        self.velocity_publisher.publish(vel_msg)
                    else:
                        vel_msg = Twist()
                        vel_msg.linear.y = -0.1
                        self.velocity_publisher.publish(vel_msg)
                elif abs(back_distance - y) > 0.01:
                    if back_distance > y:
                        vel_msg = Twist()
                        vel_msg.linear.x = -0.1
                        self.velocity_publisher.publish(vel_msg)
                    else:
                        vel_msg = Twist()
                        vel_msg.linear.x = 0.1
                        self.velocity_publisher.publish(vel_msg)
                else:
                    self.stop()
                    break
            self.rate.sleep()
        self.align_wall()
        


    """
    laser info

    header: 
    seq: 3778
    stamp: 
        secs: 1719624923
        nsecs: 443701336
    frame_id: "laser"
    angle_min: -3.1415927410125732
    angle_max: 3.1415927410125732
    angle_increment: 0.005482709966599941
    time_increment: 0.00010589129669824615
    scan_time: 0.12135142832994461
    range_min: 0.15000000596046448
    range_max: 12.0
    ranges: "<array type: float32, length: 1147>"
    intensities: "<array type: float32, length: 1147>"
    """
    def align_wall(self):
        while not rospy.is_shutdown():
            if self.laser_data:
                arr = np.array(self.laser_data.ranges)
                # sub_arr = arr[764:956]
                sub_arr = arr[800:900]
                min_index_in_sub_arr = np.argmin(sub_arr)
                left_distance_abs = sub_arr[min_index_in_sub_arr]
                left_distance_abs_index = 764 + min_index_in_sub_arr
                left_distance__para = (self.laser_data.ranges[858]+self.laser_data.ranges[859]+self.laser_data.ranges[860])/3
                # print(left_distance_abs/left_distance__para )
                if left_distance_abs == float('inf') or left_distance__para == float('inf'):
                    continue
                if left_distance_abs/left_distance__para > 1 or left_distance_abs/left_distance__para < -1:
                    continue
                print(left_distance_abs,left_distance__para)
                if left_distance_abs_index > 859:
                    error = math.acos(left_distance_abs/left_distance__para)
                else:
                    error = - math.acos(left_distance_abs/left_distance__para)
                print(error)
                if abs(error) > 0.01:
                    self.correct_orientation(error)
                else:
                    self.stop()
                    # break
            self.rate.sleep()




    def stop(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)    

    def correct_orientation(self, error):
        vel_msg = Twist()
        vel_msg.angular.z = -0.1 if error > 0 else 0.1
        self.velocity_publisher.publish(vel_msg)
                    
if __name__ == '__main__':
    init_position = initPosition()
    rospy.spin()
 


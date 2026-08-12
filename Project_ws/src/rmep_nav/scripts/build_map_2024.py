#!/usr/bin/python3
# coding=UTF-8

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction,MoveBaseGoal
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from tf.transformations import quaternion_from_euler
import math
import numpy

def go_linear(linear_speed,goal_distance):
    cmd_vel = cmd_vel = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
    target_linear_time = math.fabs(goal_distance /linear_speed)
    twist_linear = Twist()
    twist_linear.linear.x = 0
    twist_linear.linear.y = linear_speed
    twist_linear.linear.z = 0
    twist_linear.angular.x = 0
    twist_linear.angular.y = 0
    twist_linear.angular.z = 0
    go_start_time = rospy.Time.now()
    rate = rospy.Rate(100)  
    while (rospy.Time.now() - go_start_time).to_sec() < target_linear_time:
        cmd_vel.publish(twist_linear)
        rate.sleep()
    twist_linear.linear.x = 0
    twist_linear.linear.y = 0
    twist_linear.angular.z = 0
    cmd_vel.publish(twist_linear)

def turn_ang(ang_speed,goal_rotation):
    cmd_vel = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
    twist_ang = Twist()
    twist_ang.angular.z=ang_speed
    twist_ang.linear.x = 0
    twist_ang.linear.y = 0
    twist_ang.linear.z = 0
    twist_ang.angular.x = 0
    twist_ang.angular.y = 0
    target_rotation_time = math.fabs(goal_rotation/ang_speed)
    turn_start_time = rospy.Time.now()
    rate = rospy.Rate(50)  
    while (rospy.Time.now() - turn_start_time).to_sec() < target_rotation_time:
        cmd_vel.publish(twist_ang)
        rate.sleep()
    twist_ang.angular.z = 0
    cmd_vel.publish(twist_ang)



rospy.init_node('build_map', anonymous=True)
go_linear(-0.1,0.2)
# turn_ang(-0.5,1.57079633)
# go_linear(0.2,7.25)
# turn_ang(-0.5,1.57079633) 
# go_linear(0.2,2.3)
# turn_ang(-0.5,1.57079633)
# go_linear(0.2,6)


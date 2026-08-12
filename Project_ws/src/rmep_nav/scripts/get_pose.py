#!/usr/bin/python3
# coding=UTF-8

import rospy
import tf
import geometry_msgs.msg as geometry_msgs
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
import math

def get_pose():
    listener = tf.TransformListener() 
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    _1, _2, yaw3 = euler_from_quaternion(rot)
    print("x:")
    print(trans[0])
    print("y:")
    print(trans[1])
    print("yaw:")
    print(yaw3)
    print("------------")

def go_linear(linear_speed,goal_distance):
    cmd_vel = cmd_vel = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
    target_linear_time = math.fabs(goal_distance /linear_speed)
    twist_linear = Twist()
    twist_linear.linear.x = linear_speed
    twist_linear.linear.y = 0
    twist_linear.linear.z = 0
    twist_linear.angular.x = 0
    twist_linear.angular.y = 0
    twist_linear.angular.z = 0
    go_start_time = rospy.Time.now()
    rate = rospy.Rate(50)  
    while (rospy.Time.now() - go_start_time).to_sec() < target_linear_time:
        cmd_vel.publish(twist_linear)
        rate.sleep()
    twist_linear.linear.x = 0
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
go_linear(0.3,2.75)
get_pose()
turn_ang(-0.5,1.57079633)
go_linear(0.3,7.25)
get_pose()
turn_ang(-0.5,1.57079633)
go_linear(0.3,2.3)
get_pose()
turn_ang(-0.5,1.57079633)
go_linear(0.3,6)
get_pose()


#!/usr/bin/python3
# coding=UTF-8

import cv2
import rospy
import sys
import numpy as np
from pyzbar import pyzbar
from fabric2 import Connection
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge , CvBridgeError
import rospy
import actionlib
from std_msgs.msg import Bool
from move_base_msgs.msg import MoveBaseAction,MoveBaseGoal
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from tf.transformations import quaternion_from_euler,euler_from_quaternion
import math
import tf
from fabric2 import Connection
import inter

class nav:
    def __init__(self):
        self.step_line = 1.0
        self.QR_code =np.zeros(0)
        self.cv_bridge = CvBridge()
        self.camera_info_sub  = rospy.Subscriber("ep_cam/image_raw", Image, self.camera_info_cb, queue_size=30)
        self.finish_pub = rospy.Publisher('/is_finished',Bool,queue_size=1) 
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while(not self.move_base.wait_for_server(rospy.Duration(1.0))):
            continue
        self.nav_for_projecte()
    

    def go_line(self,begin,end):
        num = math.ceil(math.sqrt(math.pow(begin.target_pose.pose.position.x-end.target_pose.pose.position.x,2)+math.pow(begin.target_pose.pose.position.y-end.target_pose.pose.position.y,2)))
        goallist = list()
        yaw = 1.0
        q = Quaternion()
        x1 = begin.target_pose.pose.position.x
        y1 = begin.target_pose.pose.position.y
        x2 = end.target_pose.pose.position.x
        y2 = end.target_pose.pose.position.y
        if(x1==x2):
            if(y1>y2):
                yaw = -math.pi/2
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
                end.target_pose.pose.orientation.x = q[0]
                end.target_pose.pose.orientation.y = q[1]
                end.target_pose.pose.orientation.z = q[2]
                end.target_pose.pose.orientation.w = q[3]               
                goallist.append(begin)
                for temp in range(1,num):
                    goaltemp = MoveBaseGoal()
                    goaltemp.target_pose.pose.position.x = goallist[temp-1].target_pose.pose.position.x
                    goaltemp.target_pose.pose.position.y = goallist[temp-1].target_pose.pose.position.y - self.step_line
                    goaltemp.target_pose.pose.orientation.x = q[0]
                    goaltemp.target_pose.pose.orientation.y = q[1]
                    goaltemp.target_pose.pose.orientation.z = q[2]
                    goaltemp.target_pose.pose.orientation.w = q[3]
                    goallist.append(goaltemp)
                goallist.append(end)
            else:
                yaw = math.pi/2
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
                end.target_pose.pose.orientation.x = q[0]
                end.target_pose.pose.orientation.y = q[1]
                end.target_pose.pose.orientation.z = q[2]
                end.target_pose.pose.orientation.w = q[3]                
                goallist.append(begin)
                for temp in range(1,num):
                    goaltemp = MoveBaseGoal()
                    goaltemp.target_pose.pose.position.x = goallist[temp-1].target_pose.pose.position.x
                    goaltemp.target_pose.pose.position.y = goallist[temp-1].target_pose.pose.position.y + self.step_line
                    goaltemp.target_pose.pose.orientation.x = q[0]
                    goaltemp.target_pose.pose.orientation.y = q[1]
                    goaltemp.target_pose.pose.orientation.z = q[2]
                    goaltemp.target_pose.pose.orientation.w = q[3]
                    goallist.append(goaltemp)
                goallist.append(end)
        elif(y1==y2):
            if(x1>x2):
                yaw = math.pi
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
                end.target_pose.pose.orientation.x = q[0]
                end.target_pose.pose.orientation.y = q[1]
                end.target_pose.pose.orientation.z = q[2]
                end.target_pose.pose.orientation.w = q[3]               
                goallist.append(begin)
                for temp in range(1,num):
                    goaltemp = MoveBaseGoal()
                    goaltemp.target_pose.pose.position.x = goallist[temp-1].target_pose.pose.position.x - self.step_line
                    goaltemp.target_pose.pose.position.y = goallist[temp-1].target_pose.pose.position.y
                    goaltemp.target_pose.pose.orientation.x = q[0]
                    goaltemp.target_pose.pose.orientation.y = q[1]
                    goaltemp.target_pose.pose.orientation.z = q[2]
                    goaltemp.target_pose.pose.orientation.w = q[3]
                    goallist.append(goaltemp)
                goallist.append(end)
            else:
                yaw = 0
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
                end.target_pose.pose.orientation.x = q[0]
                end.target_pose.pose.orientation.y = q[1]
                end.target_pose.pose.orientation.z = q[2]
                end.target_pose.pose.orientation.w = q[3]               
                goallist.append(begin)
                for temp in range(1,num):
                    goaltemp = MoveBaseGoal()
                    goaltemp.target_pose.pose.position.x = goallist[temp-1].target_pose.pose.position.x + self.step_line
                    goaltemp.target_pose.pose.position.y = goallist[temp-1].target_pose.pose.position.y
                    goaltemp.target_pose.pose.orientation.x = q[0]
                    goaltemp.target_pose.pose.orientation.y = q[1]
                    goaltemp.target_pose.pose.orientation.z = q[2]
                    goaltemp.target_pose.pose.orientation.w = q[3]
                    goallist.append(goaltemp)
                goallist.append(end)
        elif((y2-y1)/(x2-x1)>0):
            k = (y2-y1)/(x2-x1)
            if(x1>x2):
                yaw = -(math.pi - math.atan(k))
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
                end.target_pose.pose.orientation.x = q[0]
                end.target_pose.pose.orientation.y = q[1]
                end.target_pose.pose.orientation.z = q[2]
                end.target_pose.pose.orientation.w = q[3]              
                goallist.append(begin)
                for temp in range(1,num):
                    goaltemp = MoveBaseGoal()
                    goaltemp.target_pose.pose.position.x = goallist[temp-1].target_pose.pose.position.x - self.step_line*math.cos(math.atan(k))
                    goaltemp.target_pose.pose.position.y = goallist[temp-1].target_pose.pose.position.y - self.step_line*math.sin(math.atan(k))
                    goaltemp.target_pose.pose.orientation.x = q[0]
                    goaltemp.target_pose.pose.orientation.y = q[1]
                    goaltemp.target_pose.pose.orientation.z = q[2]
                    goaltemp.target_pose.pose.orientation.w = q[3]
                    goallist.append(goaltemp)
                goallist.append(end)
            else:
                yaw = math.atan(k)
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
                end.target_pose.pose.orientation.x = q[0]
                end.target_pose.pose.orientation.y = q[1]
                end.target_pose.pose.orientation.z = q[2]
                end.target_pose.pose.orientation.w = q[3]
                goallist.append(begin)
                for temp in range(1,num):
                    goaltemp = MoveBaseGoal()
                    goaltemp.target_pose.pose.position.x = goallist[temp-1].target_pose.pose.position.x + self.step_line*math.cos(math.atan(k))
                    goaltemp.target_pose.pose.position.y = goallist[temp-1].target_pose.pose.position.y + self.step_line*math.sin(math.atan(k))
                    goaltemp.target_pose.pose.orientation.x = q[0]
                    goaltemp.target_pose.pose.orientation.y = q[1]
                    goaltemp.target_pose.pose.orientation.z = q[2]
                    goaltemp.target_pose.pose.orientation.w = q[3]
                    goallist.append(goaltemp)
                goallist.append(end)
        else:
            k = -(y2-y1)/(x2-x1)
            if(x1>x2):
                yaw = math.pi - math.atan(k)
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
                end.target_pose.pose.orientation.x = q[0]
                end.target_pose.pose.orientation.y = q[1]
                end.target_pose.pose.orientation.z = q[2]
                end.target_pose.pose.orientation.w = q[3]               
                goallist.append(begin)
                for temp in range(1,num):
                    goaltemp = MoveBaseGoal()
                    goaltemp.target_pose.pose.position.x = goallist[temp-1].target_pose.pose.position.x - self.step_line*math.cos(math.atan(k))
                    goaltemp.target_pose.pose.position.y = goallist[temp-1].target_pose.pose.position.y + self.step_line*math.sin(math.atan(k))
                    goaltemp.target_pose.pose.orientation.x = q[0]
                    goaltemp.target_pose.pose.orientation.y = q[1]
                    goaltemp.target_pose.pose.orientation.z = q[2]
                    goaltemp.target_pose.pose.orientation.w = q[3]
                    goallist.append(goaltemp)
                goallist.append(end)
            else:
                yaw = -math.atan(k)
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
                end.target_pose.pose.orientation.x = q[0]
                end.target_pose.pose.orientation.y = q[1]
                end.target_pose.pose.orientation.z = q[2]
                end.target_pose.pose.orientation.w = q[3]                
                goallist.append(begin)
                for temp in range(1,num):
                    goaltemp = MoveBaseGoal()
                    goaltemp.target_pose.pose.position.x = goallist[temp-1].target_pose.pose.position.x + self.step_line*math.cos(math.atan(k))
                    goaltemp.target_pose.pose.position.y = goallist[temp-1].target_pose.pose.position.y - self.step_line*math.sin(math.atan(k))
                    print(goaltemp.target_pose.pose.position.x)
                    print(goaltemp.target_pose.pose.position.y)
                    goaltemp.target_pose.pose.orientation.x = q[0]
                    goaltemp.target_pose.pose.orientation.y = q[1]
                    goaltemp.target_pose.pose.orientation.z = q[2]
                    goaltemp.target_pose.pose.orientation.w = q[3]
                    goallist.append(goaltemp)
                goallist.append(end)
        for temp in goallist:
            temp.target_pose.header.frame_id = "map"
            temp.target_pose.header.stamp = rospy.Time.now()
            self.move(temp)
        print(goallist)


    def static_line(self,begin,end):
        q = Quaternion()
        x1 = begin.target_pose.pose.position.x
        y1 = begin.target_pose.pose.position.y
        x2 = end.target_pose.pose.position.x
        y2 = end.target_pose.pose.position.y
        if(x1==x2):
            if(y1>y2):
                yaw = -math.pi/2
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]            
            else:
                yaw = math.pi/2
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
        elif(y1==y2):
            if(x1>x2):
                yaw = math.pi
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
            else:
                yaw = 0
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
        elif((y2-y1)/(x2-x1)>0):
            k = (y2-y1)/(x2-x1)
            if(x1>x2):
                yaw = -(math.pi - math.atan(k))
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]          
            else:
                yaw = math.atan(k)
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
        else:
            k = -(y2-y1)/(x2-x1)
            if(x1>x2):
                yaw = math.pi - math.atan(k)
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
            else:
                yaw = -math.atan(k)
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
        begin.target_pose.header.frame_id = "map"
        begin.target_pose.header.stamp = rospy.Time.now()
        self.move(begin)
        distance = int(math.sqrt(math.pow(begin.target_pose.pose.position.x-end.target_pose.pose.position.x,2)+math.pow(begin.target_pose.pose.position.y-end.target_pose.pose.position.y,2)))
        self.cmd_vel = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        rate = 50
        r = rospy.Rate(rate)
        linear_speed = 0.5
        goal_distance = 0.3
        linear_duration = goal_distance/linear_speed
        move_cmd = Twist()
        move_cmd.linear.x=linear_speed
        listener = tf.TransformListener() 
        for k in range(15):
            turn_start_time2 = rospy.Time.now()
            while (rospy.Time.now() - turn_start_time2).to_sec() < linear_duration:
                self.cmd_vel.publish(move_cmd)
                r.sleep()
            move_cmd2 = Twist()
            self.cmd_vel.publish(move_cmd2)
            rospy.sleep(2.5)
           
            while not rospy.is_shutdown():
                try:
                    (trans,rot) = listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            print('position',trans)
            x4 = end.target_pose.pose.position.x
            y4 = end.target_pose.pose.position.y
            x3 = trans[0]
            y3 = trans[1]
            if(x3==x4):
                if(y3>y4):
                    yaw2 = -math.pi/2
                else:
                    yaw2 = math.pi/2
            elif(y3==y4):
                if(x3>x4):
                    yaw2 = math.pi
                else:
                    yaw2 = 0
            elif((y4-y3)/(x4-x3)>0):
                k = (y4-y3)/(x4-x3)
                if(x3>x4):
                    yaw2 = -(math.pi - math.atan(k))   
                else:
                    yaw2 = math.atan(k)
            else:
                k = -(y4-y3)/(x4-x3)
                if(x3>x4):
                    yaw2 = math.pi - math.atan(k)
                else:
                    yaw2 = -math.atan(k)
            _1, _2, yaw3 = euler_from_quaternion(rot)
         
            if( math.fabs(yaw3-yaw2)>0.025):
                print("---------------")
                print(yaw3-yaw2)
                print("adjust orientation!")
                cmd_vel_ang_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
                twist_ang = Twist()
                if(yaw3-yaw2<0):
                    target_angle_radians = math.fabs(yaw3-yaw2)/2
                    twist_ang.angular.z = 0.5
                else:
                    target_angle_radians = math.fabs(yaw3-yaw2)/2
                    twist_ang.angular.z = -0.5
                twist_ang.linear.x = 0
                twist_ang.linear.y = 0
                twist_ang.linear.z = 0
                twist_ang.angular.x = 0
                twist_ang.angular.y = 0
                target_rotation_time = target_angle_radians / 0.5
                turn_start_time = rospy.Time.now()
                rate = rospy.Rate(200)  
                while (rospy.Time.now() - turn_start_time).to_sec() < target_rotation_time:
                    cmd_vel_ang_pub.publish(twist_ang)
                    rate.sleep()
                twist_ang.angular.z = 0
                cmd_vel_ang_pub.publish(twist_ang)
            else:
                continue
        #yaw = yaw+math.pi/2
        #q = quaternion_from_euler(0,0,yaw)
        end.target_pose.pose.orientation.x = q[0]
        end.target_pose.pose.orientation.y = q[1]
        end.target_pose.pose.orientation.z = q[2]
        end.target_pose.pose.orientation.w = q[3]
        end.target_pose.header.frame_id = "map"
        end.target_pose.header.stamp = rospy.Time.now()
        self.move(end)

    def static_line_back(self,begin,end):
        q = Quaternion()
        x1 = begin.target_pose.pose.position.x
        y1 = begin.target_pose.pose.position.y
        x2 = end.target_pose.pose.position.x
        y2 = end.target_pose.pose.position.y
        if(x1==x2):
            if(y1>y2):
                yaw = -math.pi/2+math.pi
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]            
            else:
                yaw = math.pi/2+math.pi
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
        elif(y1==y2):
            if(x1>x2):
                yaw = math.pi+math.pi
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
            else:
                yaw = 0+math.pi
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
        elif((y2-y1)/(x2-x1)>0):
            k = (y2-y1)/(x2-x1)
            if(x1>x2):
                yaw = -(math.pi - math.atan(k))+math.pi
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]          
            else:
                yaw = math.atan(k)+math.pi
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
        else:
            k = -(y2-y1)/(x2-x1)
            if(x1>x2):
                yaw = math.pi - math.atan(k)+math.pi
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
            else:
                yaw = -math.atan(k)+math.pi
                q = quaternion_from_euler(0,0,yaw)
                begin.target_pose.pose.orientation.x = q[0]
                begin.target_pose.pose.orientation.y = q[1]
                begin.target_pose.pose.orientation.z = q[2]
                begin.target_pose.pose.orientation.w = q[3]
        begin.target_pose.header.frame_id = "map"
        begin.target_pose.header.stamp = rospy.Time.now()
        self.move(begin)
        distance = int(math.sqrt(math.pow(begin.target_pose.pose.position.x-end.target_pose.pose.position.x,2)+math.pow(begin.target_pose.pose.position.y-end.target_pose.pose.position.y,2)))
        self.cmd_vel = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        rate = 50
        r = rospy.Rate(rate)
        linear_speed = -0.5
        goal_distance = 0.3
        linear_duration = -goal_distance/linear_speed
        move_cmd = Twist()
        move_cmd.linear.x=linear_speed
        listener = tf.TransformListener() 
        for k in range(14): 
            turn_start_time2 = rospy.Time.now()
            while (rospy.Time.now() - turn_start_time2).to_sec() < linear_duration:
                self.cmd_vel.publish(move_cmd)
                r.sleep()
            move_cmd2 = Twist()
            self.cmd_vel.publish(move_cmd2)
            rospy.sleep(2.5)
            while not rospy.is_shutdown():
                try:
                    (trans,rot) = listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            print('position',trans)
            x4 = end.target_pose.pose.position.x
            y4 = end.target_pose.pose.position.y
            x3 = trans[0]
            y3 = trans[1]
            if(x3==x4):
                if(y3>y4):
                    yaw2 = -math.pi/2
                else:
                    yaw2 = math.pi/2
            elif(y3==y4):
                if(x3>x4):
                    yaw2 = math.pi
                else:
                    yaw2 = 0
            elif((y4-y3)/(x4-x3)>0):
                k = (y4-y3)/(x4-x3)
                if(x3>x4):
                    yaw2 = -(math.pi - math.atan(k))   
                else:
                    yaw2 = math.atan(k)
            else:
                k = -(y4-y3)/(x4-x3)
                if(x3>x4):
                    yaw2 = math.pi - math.atan(k)
                else:
                    yaw2 = -math.atan(k)
            _1, _2, yaw3 = euler_from_quaternion(rot)
            yaw3 = yaw3+math.pi
            l = Quaternion()
            l = quaternion_from_euler(0,0,yaw3)
            _5,_6,yaw3 = euler_from_quaternion(l)
            j = Quaternion()
            j = quaternion_from_euler(0,0,yaw2)
            _3,_4,yaw2 = euler_from_quaternion(j)
            if( math.fabs(yaw3-yaw2)>0.025):
                cmd_vel_ang_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
                twist_ang = Twist()
                if(yaw3-yaw2<0):
                    target_angle_radians = math.fabs(yaw3-yaw2)/2
                    twist_ang.angular.z = 0.5
                else:
                    target_angle_radians = math.fabs(yaw3-yaw2)/2
                    twist_ang.angular.z = -0.5
                twist_ang.linear.x = 0
                twist_ang.linear.y = 0
                twist_ang.linear.z = 0
                twist_ang.angular.x = 0
                twist_ang.angular.y = 0
                target_rotation_time = target_angle_radians / 0.5
                turn_start_time = rospy.Time.now()
                rate = rospy.Rate(200)  
                while (rospy.Time.now() - turn_start_time).to_sec() < target_rotation_time:
                    cmd_vel_ang_pub.publish(twist_ang)
                    rate.sleep()
                twist_ang.angular.z = 0
                cmd_vel_ang_pub.publish(twist_ang)
            else:
                continue


    def move(self,goal):
        self.move_base.send_goal(goal)
        self.move_base.wait_for_result()
        if(self.move_base.get_state()==actionlib.GoalStatus.SUCCEEDED):
            rospy.loginfo("SUCCEEDED")
        else:
            rospy.loginfo("FAILED FOR SOME REASON")
    
    def nav_for_projecte(self):
        a= MoveBaseGoal()
        b= MoveBaseGoal()
        c= MoveBaseGoal()
        d= MoveBaseGoal()
        e= MoveBaseGoal()
        a.target_pose.pose.position.x=-2.0094568304544937
        a.target_pose.pose.position.y=-1.2958710435147753
        b.target_pose.pose.position.x=-5.478937464401823
        b.target_pose.pose.position.y=5.299398857985884
        c.target_pose.pose.position.x=-3.4337962509301083
        c.target_pose.pose.position.y=6.390394975730967
        d.target_pose.pose.position.x=-0.8107470101170875
        d.target_pose.pose.position.y=0.8853096241048232
        e.target_pose.pose.position.x=0
        e.target_pose.pose.position.y=0
        e.target_pose.header.frame_id = "map"
        e.target_pose.header.stamp = rospy.Time.now()
        yaw = -2.63306705178697
        q = quaternion_from_euler(0,0,yaw)
        e.target_pose.pose.orientation.x = q[0]
        e.target_pose.pose.orientation.y = q[1]
        e.target_pose.pose.orientation.z = q[2]
        e.target_pose.pose.orientation.w = q[3]
        self.is_go('/mnt/start.txt')
        self.static_line(a,b)
        self.static_line_back(c,d)
        self.move(e)
        self.has_arrive('/mnt/finish.txt')
        self.sent_code()

    def sent_code(self):
        hostname = '192.168.110.55'
        username = 'root'
        password = '123456'
        filepath = '/mnt/tta/list_of_goods_fromCAR.txt'
        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})
        print("connect suc!!")
        b = inter.create_file(conn,filepath)
        print(b)
        print("test_opr")
        count = 0   
        while count<np.size(self.QR_code):
            try:
                a = inter.file_echo(conn,filepath,str(self.QR_code[count]))
                count+=1  
            except Exception as e:
                continue
        conn.close()
        print(self.QR_code)
    

    def is_go(self,path):
        hostname = '192.168.110.55'
        username = 'root'
        password = '123456'
        filepath = path

        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})

        while True:
            try:
                result = conn.run('test -e {}'.format(filepath), hide=True)
                if result.ok: 
                    conn.run(f'rm {filepath}')
                    conn.close()
                    break
            except Exception as e:
                print("continue to check")
                continue
    
    def has_arrive(self,path):
        hostname = '192.168.110.55'
        username = 'root'
        password = '123456'
        filepath = path

        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})

        conn.run(f"touch {filepath}")

    def QR_Scan(self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        decoded = pyzbar.decode(gray)
        Code = np.zeros(0)
        for barcode in decoded:
            (x,y,w,h) = barcode.rect
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            Data = barcode.data.decode("utf-8")
            codetext = "{}".format(Data)
            Code = np.append(Code,codetext)
        return Code
    
    

    def camera_info_cb(self,msg):
        

        image = self.cv_bridge.imgmsg_to_cv2(msg,"bgr8")
        #print(image)
        ImgCode = self.QR_Scan(image)

        for temp in range(0,max(0,np.size(ImgCode))):
            Judge = True
            #print("test1")

            for real in range(0,max(0,np.size(self.QR_code))):
                if(ImgCode[temp]==self.QR_code[real]):
                    #print("test2")
                    Judge = False


            if(Judge):
                #print("test3")
                self.QR_code = np.append(self.QR_code,ImgCode[temp])

                print(self.QR_code)

        if(np.size(self.QR_code)!=0):
            pass
            #print(self.QR_code)

        cv2.imshow("Robot",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return




    


if __name__ == '__main__':  
    
    rospy.init_node('rmep_nav', anonymous=True)
    rmep_nav = nav()
    rospy.spin()
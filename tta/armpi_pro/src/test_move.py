#!/usr/bin/python3
# coding=utf8
import sys
import rospy
from chassis_control.msg import *

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
    
print('''
**********************************************************
***************功能:小车前后左右移动例程****************
**********************************************************
----------------------------------------------------------
Official website:https://www.hiwonder.com
Online mall:https://hiwonder.tmall.com
----------------------------------------------------------
Tips:
 * 按下Ctrl+C可关闭此次程序运行，若失败请多次尝试！
----------------------------------------------------------
''')

start = True
#关闭前处理
def stop():
    global start

    start = False
    print('关闭中...')
    set_velocity.publish(0,0,0)  # 发布底盘控制消息,停止移动
    
if __name__ == '__main__':
    # 初始化节点
    rospy.init_node('lab1', log_level=rospy.DEBUG)
    rospy.on_shutdown(stop)
    # 麦轮底盘控制
    set_velocity = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)
    

    rospy.sleep(1)

    set_velocity.publish(60,90,0)
    rospy.sleep(2)  
    set_velocity.publish(60,270,0) # 
    rospy.sleep(2) # 延时2秒
    set_velocity.publish(60,180,0) # 
    rospy.sleep(2) # 延时2秒
    set_velocity.publish(60,270,0) # 
    rospy.sleep(2) # 延时2秒
    set_velocity.publish(60,0,0) # 
    rospy.sleep(2)
    set_velocity.publish(10,270,0)
    rospy.sleep(3)
    set_velocity.publish(10,135,0)
    rospy.sleep(3)
    set_velocity.publish(10,270,1.257)
    rospy.sleep(4)

    set_velocity.publish(0,0,0)  # 发布底盘控制消息,停止移动
    print('已关闭')

        



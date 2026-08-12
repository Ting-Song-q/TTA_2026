#!/usr/bin/python3
# coding=UTF-8

import rospy
from geometry_msgs.msg import Twist

def move_in_circle():
    # ³õÊ¼»¯ ROS ½Úµã
    rospy.init_node('circle_mover', anonymous=True)

    # ´´½¨Ò»¸ö publisher ¶ÔÏó£¬·¢²¼µ½ /cmd_vel »°Ìâ
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Éè¶¨ÔË¶¯µÄËÙ¶È
    linear_speed = 0.2  # ¿ÉÒÔµ÷ÕûÕâ¸öËÙ¶ÈÖµ
    radius = 0.5
    angular_speed = linear_speed / radius

    # ¼ÆËãÍê³ÉÒ»È¦ËùÐèµÄÊ±¼ä
    circumference = 2 * 3.14159 * radius
    duration = circumference / linear_speed

    # ´´½¨Ò»¸ö Twist ÏûÏ¢
    twist = Twist()
    twist.linear.x = linear_speed
    twist.angular.z = angular_speed

    # ÉèÖÃ¶¨Ê±Æ÷Í£Ö¹Ð¡³µ
    def stop_robot(event):
        twist.linear.x = 0
        twist.angular.z = 0
        cmd_vel_pub.publish(twist)

    # ·¢²¼Ô²ÖÜÔË¶¯µÄËÙ¶ÈÖ¸Áî
    rate = rospy.Rate(10)  # 10 Hz
    rospy.Timer(rospy.Duration(duration), stop_robot, oneshot=True)

    while not rospy.is_shutdown():
        cmd_vel_pub.publish(twist)
        rate.sleep()

if __name__ == '__main__':
    try:
        move_in_circle()
    except rospy.ROSInterruptException:
        pass

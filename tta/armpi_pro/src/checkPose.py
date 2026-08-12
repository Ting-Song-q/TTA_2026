#!/usr/bin/python3
# coding=utf8
import sys
import rospy
import time
from armpi_pro import bus_servo_control
from kinematics import ik_transform
from hiwonder_servo_msgs.msg import MultiRawIdPosDur

def check_initial_pose():
    rospy.init_node('check_pose')
    joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
    ik = ik_transform.ArmIK()
    
    rospy.sleep(0.5)
    rospy.loginfo("Setting Initial Pose...")
    
    # Target: X=0, Y=0.15m, Z=0.10m, Pitch=-180
    target = ik.setPitchRanges((0, 0.15, 0.10), -200, -200, 0)
    
    if target:
        servo_data = target[1]
        bus_servo_control.set_servos(joints_pub, 1500, (
            (1, 200), (2, 500), 
            (3, servo_data['servo3']), (4, servo_data['servo4']),
            (5, servo_data['servo5']), (6, servo_data['servo6'])
        ))
        rospy.loginfo("Pose Set: X=0, Y=0.15, Z=0.10, Pitch=-200")
    else:
        rospy.logerr("IK Solution Not Found for Initial Pose!")

if __name__ == '__main__':
    try:
        check_initial_pose()
    except rospy.ROSInterruptException:
        pass


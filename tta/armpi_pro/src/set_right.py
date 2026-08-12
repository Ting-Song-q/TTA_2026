#!/usr/bin/python3
# coding=utf8
import sys
import rospy
from hiwonder_servo_msgs.msg import MultiRawIdPosDur
from armpi_pro import bus_servo_control
from kinematics import ik_transform

def set_arm_right():
    # 初始化节点
    rospy.init_node('set_arm_right', anonymous=True)
    
    # 关节发布者
    joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
    
    # IK解算器
    ik = ik_transform.ArmIK()
    
    # 等待连接
    rospy.sleep(0.5)
    
    rospy.loginfo("正在将机械臂移动到正右侧 (ID 6 = 250)...")
    
    # 计算机械臂的基础姿态 (使用与reset_arm相同的参数)
    # setPitchRanges((x, y, z), alpha, alpha1, alpha2)
    # x=0, y=0.15, z=0.10 是一个标准的待机姿态
    target = ik.setPitchRanges((0, 0.15, 0.10), -180, -180, 0)
    
    if target:
        servo_data = target[1]
        # 设置舵机位置
        # ID 6 设置为 250 (正右)
        # ID 1 (爪子) 设置为 200 (闭合/待机)
        # ID 2 (云台) 设置为 500 (居中)
        bus_servo_control.set_servos(joints_pub, 1500, (
            (1, 200),               
            (2, 500),               
            (3, servo_data['servo3']),
            (4, servo_data['servo4']),
            (5, servo_data['servo5']),
            (6, 250)                # 强制底座转向正右
        ))
        rospy.loginfo("指令已发送。")
    else:
        rospy.logerr("IK 解算失败！")
        
    rospy.sleep(2.0)

if __name__ == '__main__':
    try:
        set_arm_right()
    except rospy.ROSInterruptException:
        pass


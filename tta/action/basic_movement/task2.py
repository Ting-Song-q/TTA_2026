import sys
from time import sleep

import rospy
from chassis_control.msg import *
from std_msgs.msg import Bool

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

print('''
**********************************************************
********************功能:小车前进例程********************
**********************************************************
----------------------------------------------------------
Official website:https://www.hiwonder.com
Online mall:https://hiwonder.tmall.com
----------------------------------------------------------
Tips:
 * 按下Ctrl+C可关闭此次程序运行，若失败请多次尝试！
----------------------------------------------------------
''')


def forward(set, interval=1, vel=60):
    set.publish(vel, 90, 0)
    rospy.sleep(interval)
    print("前进")

def slant(set,angle,interval=1,vel=80):
    set.publish(vel,angle,0)
    rospy.sleep(interval)
    print(f"以{angle}角度斜向前进")

def U_turn(set):
    for i in range(2):
        set.publish(0, 90, -0.3)  # 顺时针旋转
        if i==0:
            rospy.sleep(0.5)
        else:
            rospy.sleep(7)


start = True


# 关闭前处理
def stop():
    global start

    start = False
    print('关闭中...')
    set_velocity.publish(0, 0, 0)  # 发布底盘控制消息,停止移动


if __name__ == '__main__':
    # 初始化节点
    rospy.init_node('car_forward_demo', log_level=rospy.DEBUG)
    rospy.on_shutdown(stop)
    # 麦轮底盘控制
    set_velocity = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)

    #前进一段距离，从起点H到达一区
    for i in range(5):
        forward(set_velocity, 2, 60)
        if not start:
            break
    print("前进完毕，到达一区，车机开始分离")
    #阻塞，等待车机交互完成
    
    forward(set_velocity,2,0)
    rospy.wait_for_message('/uav_signal', Bool)
    print("收到无人机信号，继续前往二区")
    
    #forward(set_velocity,2,0)
    #sleep(2)
    print("小车开始前往二区")
    #斜向运动，从一区到二区
    for i in range(10):
        slant(set_velocity,45)
        if not start:
            break
            
    print("小车到达二区,等待飞机降落")
    forward(set_velocity,2,0)
    rospy.wait_for_message('/uav_signal', Bool)
    print("飞机降落完成，小车开始前往充电区")
    #forward(set_velocity, 2, 0)
    #ssleep(2)

    #从二区前往充电区
    for i in range(10):
        slant(set_velocity,135)
        if not start:
            break
    print("到达充电区，开始充电")
    forward(set_velocity, 2, 0)
    sleep(5)
    
    U_turn(set_velocity)
    print("掉头转向成功,开始返回")
    for i in range(10):
        forward(set_velocity,2,60)
        if not start:
            break
    print("返回成功")
    set_velocity.publish(0, 0, 0)  # 发布底盘控制消息,停止移动
    print('已关闭')

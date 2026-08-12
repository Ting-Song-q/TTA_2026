import sys
from time import sleep
import rospy
from chassis_control.msg import *
from std_msgs.msg import Bool, String

if sys.version_info.major == 2:
    print('请使用Python3运行此程序!')
    sys.exit(0)

print('''
******************************************************************************
功能: 小车与无人机协同任务
******************************************************************************
''')


def forward(set, vel=60):
    '''直线前进，vel代表轮胎线速度'''
    set.publish(vel, 90, 0)
    rospy.sleep(1)
    print("前进")


def slant(set, angle, vel=80):
    '''呈一定角度倾斜前进'''
    set.publish(vel, angle, 0)
    rospy.sleep(1)
    print(f"以{angle}角度斜向前进")


def U_turn(set,u_time=7):
    '''掉头'''
    for i in range(2):
        set.publish(0, 90, -0.31)  # 顺时针旋转
        if i == 0:
            rospy.sleep(0.5)
        else:
            rospy.sleep(u_time)


start = True


def stop():
    global start
    start = False
    print('关闭中...')
    set_velocity.publish(0, 0, 0)


if __name__ == '__main__':
    #参数初始化
    time_H_to_1 = 20#起点到一区运动时间
    vel_H_to_1 = 100.2#起点到一区运动线速度
    angle_H_to_1 =109.0#起点到一区运动角度（正右方为0，正前方为90）
    time_1_to_2 = 22#一区到二区运动时间
    vel_1_to_2 = 102.6#一区到二区运动速度
    angle_1_to_2 = 57.0#一区到二区运动角度
    time_2_to_power = 16#二区到充电区运动时间
    vel_2_to_power = 97#二区到充电区运动速度
    angle_2_to_power = 114#二区到充电区运动角度
    time_power_sleep = 5 #充电时间
    
    r_time_H_to_1 = 13#充电区到二区时间
    r_vel_H_to_1 = 200#充电区到二区线速度
    r_angle_H_to_1 =95#充电区到二区角度
    r_time_1_to_2 = 12#二区到一区时间
    r_vel_1_to_2 = 200#二区到一区速度
    r_angle_1_to_2 = 84#二区到一区运动角度


    r_time_2_to_power = 16#一区到起点运动时间
    r_vel_2_to_power = 200#一区到起点运动速度
    r_angle_2_to_power = 117#一区到起点运动角度


    time_return=20#掉头后时间
    vel_return = 260#掉头后速度

    # 初始化节点
    rospy.init_node('car_controller', log_level=rospy.DEBUG)
    rospy.on_shutdown(stop)

    # 创建发布者，用于通知无人机
    car_status_pub = rospy.Publisher('/car_status', String, queue_size=1)

    # 麦轮底盘控制
    set_velocity = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)

    # 阶段1: 从起点H到达一区

    print("开始从起点H前往一区")
    for i in range(time_H_to_1):
        slant(set_velocity, angle_H_to_1, vel_H_to_1)
        if not start:
            break

    forward(set_velocity, 0)
    print("前进完毕，到达一区")

    # 转发消息已到达一区
    car_status_pub.publish("arrived_at_area1")
    print("向无人机发送允许起飞信号...")

    # 等待无人机起飞完成信号
    # rospy.wait_for_message('/go_area2', Bool)
    # print("收到无人机起飞完成信号，继续前往二区")

    # 阶段2: 从一区到二区
    print("开始从一区前往二区")
    for i in range(time_1_to_2):
        slant(set_velocity, angle_1_to_2, vel_1_to_2)
        if not start:
            break

    forward(set_velocity, 0)
    print("到达二区")

    # # 通知无人机已到达二区
    # car_status_pub.publish("arrived_at_area2")
    # print("等待无人机降落...")

    # 等待无人机降落完成信号
    # rospy.wait_for_message('/go_charge', Bool)
    # print("收到无人机降落完成信号，开始前往充电区")

    # 阶段3: 从二区到充电区
    print("开始从二区前往充电区")
    for i in range(time_2_to_power):
        slant(set_velocity, angle_2_to_power, vel_2_to_power)
        if not start:
            break

    print("到达充电区，开始充电")
    forward(set_velocity, 0)
    rospy.sleep(5)

    # 阶段4: 掉头返回
    U_turn(set_velocity)
    print("掉头转向成功,开始返回")

    #for i in range(time_return):
        #forward(set_velocity, vel_return)
        #if not start:
            #break
    print("开始从充电区前往二区")
    for i in range(r_time_H_to_1):
        slant(set_velocity, r_angle_H_to_1, r_vel_H_to_1)
        if not start:
            break

    forward(set_velocity, 0)

    print("开始从二区前往出发区")
    for i in range(r_time_1_to_2):
        slant(set_velocity, r_angle_1_to_2, r_vel_1_to_2)
        if not start:
            break

    forward(set_velocity, 0)

    print("返回成功")
    set_velocity.publish(0, 0, 0)
    print('任务完成，已停止')


    

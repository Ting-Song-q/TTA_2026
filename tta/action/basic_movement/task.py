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

HEARTBEAT_TIMEOUT = 3  # 心跳超时时间(秒)
last_heartbeat_time = 0


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


def U_turn(set, u_time=7):
    '''掉头'''
    for i in range(2):
        set.publish(0, 90, -0.298)  # 顺时针旋转
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
    print('小车已停止运动')


def heartbeat_callback(msg):
    global last_heartbeat_time
    last_heartbeat_time = rospy.get_time()


def emergency_stop_callback(msg):
    if msg.data:
        print("收到紧急停止信号!")
        stop()
        rospy.signal_shutdown("紧急停止")


def check_heartbeat():
    global last_heartbeat_time
    current_time = rospy.get_time()

    if current_time - last_heartbeat_time > HEARTBEAT_TIMEOUT:
        print(f"心跳超时({HEARTBEAT_TIMEOUT}秒)，无人机可能已断开连接!")
        stop()
        rospy.signal_shutdown("心跳超时")


if __name__ == '__main__':
    # 参数初始化
    time_H_to_1 = 20  # 起点到一区运动时间
    vel_H_to_1 = 100.2  # 起点到一区运动线速度
    angle_H_to_1 = 109.5  # 起点到一区运动角度（正右方为0，正前方为90）
    time_1_to_2 = 23  # 一区到二区运动时间
    vel_1_to_2 = 100  # 一区到二区运动速度
    angle_1_to_2 = 50.5  # 一区到二区运动角度
    time_2_to_power = 15  # 二区到充电区运动时间
    vel_2_to_power = 100  # 二区到充电区运动时间
    angle_2_to_power = 120  # 二区到充电区运动角度
    time_power_sleep = 5  # 充电时间

    time_return = 35  # 掉头后时间
    vel_return = 130  # 掉头后速度

    # 初始化节点
    rospy.init_node('car_uav_coordination', log_level=rospy.DEBUG)
    rospy.on_shutdown(stop)

    # 创建发布者，用于通知无人机
    car_status_pub = rospy.Publisher('/car_status', String, queue_size=1)

    # 麦轮底盘控制
    set_velocity = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)

    # 订阅心跳话题
    rospy.Subscriber('/heartbeat', Bool, heartbeat_callback)

    # 订阅紧急停止话题
    rospy.Subscriber('/emergency_stop', Bool, emergency_stop_callback)

    # 初始化心跳时间
    last_heartbeat_time = rospy.get_time()

    # 创建定时器检查心跳
    rospy.Timer(rospy.Duration(1), lambda event: check_heartbeat())

    # 阶段1: 从起点H到达一区
    print("开始从起点H前往一区")
    for i in range(time_H_to_1):
        if not start or rospy.is_shutdown():
            break
        slant(set_velocity, angle_H_to_1, vel_H_to_1)
        rospy.sleep(1)

    forward(set_velocity, 0)
    print("前进完毕，到达一区")

    # 通知无人机已到达一区
    car_status_pub.publish("arrived_at_area1")
    print("向无人机发送允许起飞信号...")

    # 等待无人机起飞完成信号
    rospy.wait_for_message('/uav_signal', Bool)
    print("收到无人机起飞完成信号，继续前往二区")

    # 阶段2: 从一区到二区
    print("开始从一区前往二区")
    for i in range(time_1_to_2):
        if not start or rospy.is_shutdown():
            break
        slant(set_velocity, angle_1_to_2, vel_1_to_2)
        rospy.sleep(1)

    forward(set_velocity, 0)
    print("到达二区")

    # 通知无人机已到达二区
    car_status_pub.publish("arrived_at_area2")
    print("等待无人机降落...")

    # 等待无人机降落完成信号
    rospy.wait_for_message('/uav_signal', Bool)
    print("收到无人机降落完成信号，开始前往充电区")

    # 阶段3: 从二区到充电区
    print("开始从二区前往充电区")
    for i in range(time_2_to_power):
        if not start or rospy.is_shutdown():
            break
        slant(set_velocity, angle_2_to_power, vel_2_to_power)
        rospy.sleep(1)

    print("到达充电区，开始充电")
    forward(set_velocity, 0)
    rospy.sleep(time_power_sleep)

    # 阶段4: 掉头返回
    U_turn(set_velocity)
    print("掉头转向成功,开始返回")

    for i in range(time_return):
        if not start or rospy.is_shutdown():
            break
        forward(set_velocity, vel_return)
        rospy.sleep(1)

    print("返回成功")
    set_velocity.publish(0, 0, 0)
    print('任务完成，已停止')
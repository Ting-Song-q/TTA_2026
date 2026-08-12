#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool, String

"""

该脚本与 car_controller.py 并行：
1. 监听 /car_status，小车到达一区后发布 /takeoff_begin
2. 等待 /takeoff_done，触发 car_controller 前往二区逻辑
3. 等待 /landing_done，触发 car_controller 前往充电区逻辑
"""

def main():
    # 初始化 ROS 节点
    rospy.init_node('car_uav_comm_local')

    # 发布：通知无人机起飞
    pub_takeoff_begin = rospy.Publisher('/takeoff_begin', Bool, queue_size=1)
    # 发布：通知 car_controller 前往二区
    pub_go_area2 = rospy.Publisher('/go_area2', Bool, queue_size=1)
    # 发布：通知 car_controller 前往充电区
    pub_go_charge = rospy.Publisher('/go_charge', Bool, queue_size=1)

    # 1. 等待 car_controller 发布的小车到达一区状态
    rospy.loginfo("[Step 1] 等待 /car_status: 小车到达一区")
    msg = rospy.wait_for_message('/car_status', String)
    if msg.data == 'arrived_at_area1':
        rospy.loginfo("小车已到达一区，发布 /takeoff_begin")
        pub_takeoff_begin.publish(True)#向飞机发送起飞指令

    # 2. 等待无人机起飞完成
    rospy.loginfo("[Step 2] 等待 /takeoff_down: 无人机起飞完成")
    rospy.wait_for_message('/takeoff_down', Bool)
    rospy.loginfo("收到 /takeoff_done，发布 /go_area2")
    pub_go_area2.publish(True)

    # 3. 等待无人机降落完成
    rospy.loginfo("[Step 3] 等待 /landing_done: 无人机降落完成")
    rospy.wait_for_message('/landing_done', Bool)
    rospy.loginfo("收到 /landing_done，发布 /go_charge")
    pub_go_charge.publish(True)

    rospy.loginfo("本地通信与控制流程完成。")

if __name__ == '__main__':
    main()

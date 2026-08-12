import time
import socket
import paramiko
import rospy
from std_msgs.msg import Bool, String

# —— 配置区 —— #
CAR_IP = '192.168.149.1'  # 小车热点 IP
CAR_USER = 'ubuntu'  # 小车用户名
CAR_PASS = 'hiwonder'  # 密码登录
ROS_PORT = 11311  # roscore 监听端口
SSH_PORT = 22


# —— Helper：检测 roscore 是否 ready —— #
def wait_for_roscore(host, port, timeout=30):
    print(f"等待 ROS Master ({host}:{port}) 就绪...", end='', flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = socket.socket()
        s.settimeout(1.0)
        try:
            s.connect((host, port))
            s.close()
            print(" ")
            return True
        except Exception:
            print('.', end='', flush=True)
            time.sleep(1)
    print(" [超时]")
    return False


def main():
    # 1. SSH 连接
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[1] SSH 连接到 {CAR_IP} …")
    ssh.connect(CAR_IP, port=SSH_PORT, username=CAR_USER, password=CAR_PASS)

    # 2. 启动 roscore
    cmd_roscore = (
        f'export ROS_MASTER_URI=http://{CAR_IP}:11311 && '
        f'export ROS_HOSTNAME={CAR_IP} && '
        'nohup roscore > /tmp/roscore.log 2>&1 &'
    )
    print("[2] 在小车上启动 roscore …")
    ssh.exec_command(cmd_roscore)

    # 3. 等待 Master 就绪
    if not wait_for_roscore(CAR_IP, ROS_PORT, timeout=20):
        print("Error: roscore 启动失败或超时，退出。")
        return

    # 初始化ROS节点
    rospy.init_node('uav_controller')
    # 创建发布者，用于向小车发送信号
    uav_signal_pub = rospy.Publisher('/uav_signal', Bool, queue_size=10)

    # 4. 启动 car_controller.py (小车控制脚本)
    cmd_demo = (
        f'export ROS_MASTER_URI=http://{CAR_IP}:11311 && '
        f'export ROS_HOSTNAME={CAR_IP} && '
        'nohup bash -lc "'
        'source /opt/ros/melodic/setup.bash && '
        'source /home/ubuntu/armpi_pro/devel/setup.bash && '
        'python3 /home/ubuntu/Desktop/basic_movements/car_controller.py'
        '" > /tmp/car_demo.log 2>&1 &'
    )

    print("[3] 在小车上启动 car_controller.py …")
    ssh.exec_command(cmd_demo)

    # 5. 等待小车到达一区
    print("[4] 等待小车到达一区...")
    rospy.wait_for_message('/car_status', String)  # 第一次调用会等待消息到来

    # 模拟起飞过程
    print("飞机开始起飞")
    input("按Enter键确认起飞成功...")

    # 向小车发送起飞完成信号
    print("起飞成功，向小车发送可前进去往二区信号")
    uav_signal_pub.publish(True)

    # 6. 等待小车到达二区
    print("等待小车到达二区...")
    msg = rospy.wait_for_message('/car_status', String)
    while msg.data != "arrived_at_area2":  # 确保收到的是到达二区的消息
        msg = rospy.wait_for_message('/car_status', String)

    # 模拟降落过程
    print("开始降落...")
    input("按Enter键确认降落成功...")

    # 向小车发送降落完成信号
    uav_signal_pub.publish(True)
    print("已发送降落完成信号，小车可以前往充电区")

    print("全部流程执行完毕。")
    ssh.close()


if __name__ == "__main__":
    main()
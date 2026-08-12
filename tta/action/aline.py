#!/usr/bin/python3
# coding=utf8
"""aline：ArmPi Pro 直行测试（排查四轮转速是否一致 / 是否跑偏）。

方向约定（与 basic_movement/car_controller.py 一致）：
  正右方 = 0°，正前方 = 90°，正左方 = 180°，正后方 = 270°
  因此前进必须用 direction=90，不是 0。
  （chassis_control 内部会对 velocity 取反；direction=0 实际会表现为左移/右移一侧）

前置：
  1) source armpi_pro 工作空间
  2) chassis_control 节点已启动

用法：
  python3 aline.py
  python3 aline.py --speed 60 --duration 3
  python3 aline.py --direction 90         # 前进（默认）
  python3 aline.py --direction 0          # 侧移（标定用）
  python3 aline.py --roundtrip            # 前进 → 停 → 后退
  python3 aline.py --hang                 # 悬空观察提示

观察要点：
  - 悬空：四轮转速是否接近（左后是否明显更快）
  - 落地：轨迹是否直线、是否向一侧偏
"""

from __future__ import print_function

import argparse
import sys

import rospy
from chassis_control.msg import SetVelocity


def publish_stop(pub, times=5, interval=0.05):
    for _ in range(times):
        pub.publish(SetVelocity(velocity=0.0, direction=0.0, angular=0.0))
        rospy.sleep(interval)


def drive_once(pub, speed, direction, duration, label):
    rospy.loginfo(
        "[aline] %s: speed=%.1f mm/s  direction=%.1f°  angular=0  duration=%.2fs",
        label,
        speed,
        direction,
        duration,
    )
    pub.publish(
        SetVelocity(
            velocity=float(speed),
            direction=float(direction),
            angular=0.0,
        )
    )
    rospy.sleep(float(duration))
    publish_stop(pub)


def opposite_direction(direction_deg):
    return (float(direction_deg) + 180.0) % 360.0


def main():
    if sys.version_info.major == 2:
        print("Please run this program with python3!")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="ArmPi Pro 直行测试")
    parser.add_argument(
        "--speed",
        type=float,
        default=60.0,
        help="平移速度 mm/s（默认 60，官方 demo 常用 60~100）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="单段直行时间秒（默认 3）",
    )
    parser.add_argument(
        "--direction",
        type=float,
        default=90.0,
        help="运动方向角°：90=前进（默认），0=右，180=左，270=后",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/chassis_control/set_velocity",
        help="底盘话题",
    )
    parser.add_argument(
        "--roundtrip",
        action="store_true",
        help="前进后再沿反方向退回",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="往返中间停顿秒（仅 --roundtrip）",
    )
    parser.add_argument(
        "--hang",
        action="store_true",
        help="打印悬空测试提示后继续跑（不改变控制逻辑）",
    )
    args = parser.parse_args()

    print(
        "\n".join(
            [
                "========== aline 直行测试 ==========",
                "speed=%.1f mm/s  direction=%.1f°  duration=%.2fs"
                % (args.speed, args.direction, args.duration),
                "Ctrl+C 随时停车",
                "====================================",
            ]
        )
    )
    if args.hang:
        print(
            "[提示] 建议先悬空观察四轮转速是否一致，"
            "再落地看是否直线；左后若悬空也更快，优先查线序/驱动。"
        )

    rospy.init_node("aline_straight_test", anonymous=True)
    pub = rospy.Publisher(args.topic, SetVelocity, queue_size=1)
    rospy.on_shutdown(lambda: publish_stop(pub))
    rospy.sleep(0.5)

    try:
        drive_once(pub, args.speed, args.direction, args.duration, "forward")
        if args.roundtrip:
            if args.pause > 0:
                rospy.sleep(args.pause)
            back_dir = opposite_direction(args.direction)
            drive_once(pub, args.speed, back_dir, args.duration, "backward")
    except rospy.ROSInterruptException:
        pass
    finally:
        publish_stop(pub)
        rospy.loginfo("[aline] stopped")


if __name__ == "__main__":
    main()

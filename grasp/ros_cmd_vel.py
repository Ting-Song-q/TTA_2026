#!/usr/bin/python3
"""System-Python ROS helper that publishes one timed /cmd_vel command."""

from __future__ import print_function

import argparse

import rospy
from geometry_msgs.msg import Twist


def main():
    parser = argparse.ArgumentParser(description="Publish one timed Twist command")
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--label", default="cmd_vel")
    args = parser.parse_args()

    rospy.init_node("mobile_grasp_cmd", anonymous=True, disable_signals=True)
    publisher = rospy.Publisher(args.topic, Twist, queue_size=10)
    rate = rospy.Rate(max(1.0, args.rate))
    command = Twist()
    command.linear.x = args.vx
    command.linear.y = args.vy
    rospy.sleep(0.1)
    connections = publisher.get_num_connections()
    print("[ros-cmd] topic=%s subscribers=%d" % (args.topic, connections), flush=True)
    if connections == 0:
        print("[ros-cmd] WARNING: no subscriber is connected to the command topic", flush=True)
    print("[ros-cmd] %s vx=%+.3f vy=%+.3f t=%.2fs" % (args.label, args.vx, args.vy, args.duration), flush=True)
    try:
        started = rospy.Time.now().to_sec()
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() - started < max(0.0, args.duration):
            publisher.publish(command)
            rate.sleep()
    finally:
        publisher.publish(Twist())
        print("[ros-cmd] stopped", flush=True)


if __name__ == "__main__":
    main()

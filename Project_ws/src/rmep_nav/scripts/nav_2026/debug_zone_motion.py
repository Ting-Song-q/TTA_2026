#!/usr/bin/python3
# coding=UTF-8
"""Execute one tiny guarded base-frame movement to verify axis directions."""

import argparse
import sys

import rospy

from nav_rescue_2026 import Nav


DIRECTIONS = {
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
}


def main():
    parser = argparse.ArgumentParser(description="Guarded 3-5cm direction test")
    parser.add_argument("direction", choices=sorted(DIRECTIONS))
    parser.add_argument("--distance", type=float, default=0.03)
    parser.add_argument("--confirm", choices=("MOVE",), required=True)
    args = parser.parse_args()
    if not 0.01 <= args.distance <= 0.05:
        parser.error("--distance must be between 0.01m and 0.05m")

    nav = Nav(autostart=False, wait_for_move_base=False)
    rospy.sleep(1.0)
    if nav.laser_data is None:
        rospy.logerr("/scan unavailable; refusing to move")
        return 1

    unit_x, unit_y = DIRECTIONS[args.direction]
    step = (unit_x * args.distance, unit_y * args.distance)
    config = dict(nav.config.get("zone_entry", {}).get("adjustment", {}))
    config["linear_speed"] = min(float(config.get("linear_speed", 0.05)), 0.03)
    rospy.logwarn(
        "direction test starts in 3 seconds: %s %.3fm", args.direction, args.distance
    )
    rospy.sleep(3.0)
    success = nav._drive_zone_adjustment(step, config)
    nav.stop()
    if not success:
        rospy.logerr("direction test stopped by safety guard")
        return 1
    rospy.loginfo("direction test completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/python3
# coding=UTF-8

# BEGIN added: standalone real-vehicle zone validation tool
import argparse
import json

import rospy

from nav_rescue_2026 import Nav


def _check(nav, source_name, zone_name, zone_id):
    if source_name == "vision":
        result = nav._assert_wheels_in_zone_by_vision(zone_name, zone_id)
        return result or nav._zone_result(
            "vision",
            False,
            False,
            "vision_unavailable",
            zone_name,
            zone_id,
        )
    if source_name == "map":
        return nav._assert_wheels_in_zone_by_map(zone_name, zone_id)
    return nav.check_wheels_in_zone(zone_name, zone_id)


def main():
    parser = argparse.ArgumentParser(
        description="Check wheel containment without starting the rescue mission."
    )
    parser.add_argument("zone", choices=("parking", "pickup", "loading", "rescue"))
    parser.add_argument("--zone-id", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--source", choices=("auto", "vision", "map"), default="auto")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--adjust", action="store_true")
    parser.add_argument("--confirm", choices=("MOVE",))
    args = parser.parse_args()
    if args.zone == "rescue" and args.zone_id is None:
        parser.error("--zone-id is required for rescue")
    if args.adjust and args.confirm != "MOVE":
        parser.error("--adjust requires --confirm MOVE")
    if args.adjust and args.source == "vision":
        parser.error("--adjust uses auto fallback; --source vision is not supported")

    nav = Nav(autostart=False, wait_for_move_base=False)
    rospy.sleep(1.0)
    rate = rospy.Rate(max(args.rate, 0.1))
    while not rospy.is_shutdown():
        if args.adjust:
            passed = nav.ensure_wheels_in_zone(args.zone, args.zone_id)
            result = dict(nav._last_zone_check or {})
            result["adjustment_passed"] = bool(passed)
        else:
            result = _check(nav, args.source, args.zone, args.zone_id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if args.once or args.adjust:
            break
        rate.sleep()


if __name__ == "__main__":
    main()
# END added: standalone real-vehicle zone validation tool

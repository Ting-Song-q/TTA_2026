#!/usr/bin/python3
# coding=UTF-8
"""Read-only ROS and calibration checks before zone-entry field tests."""

import argparse
import sys
from pathlib import Path

import actionlib
import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction
from sensor_msgs.msg import Image, LaserScan

from config_loader import calibration_has_zone, load_mission_config
from zone_health import covariance_std


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "mission_config.yaml"


def _result(name, passed, detail):
    prefix = "OK" if passed else "FAIL"
    print("[%s] %-18s %s" % (prefix, name, detail))
    return passed


def _wait_message(topic, message_type, timeout):
    try:
        return rospy.wait_for_message(topic, message_type, timeout=timeout), None
    except rospy.ROSException as exc:
        return None, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Run non-moving zone test checks")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--require-camera", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    config = load_mission_config(CONFIG_PATH)
    rospy.init_node("preflight_zone_test", anonymous=True)
    checks = []

    scan, scan_error = _wait_message("/scan", LaserScan, args.timeout)
    checks.append(_result("laser /scan", scan is not None, scan_error or "fresh message"))

    amcl, amcl_error = _wait_message(
        "/amcl_pose", PoseWithCovarianceStamped, args.timeout
    )
    if amcl is None:
        checks.append(_result("AMCL /amcl_pose", False, amcl_error))
    else:
        position_std, yaw_std = covariance_std(amcl.pose.covariance)
        limits = config.get("zone_entry", {})
        healthy = (
            position_std <= float(limits.get("max_position_std", 0.15))
            and yaw_std <= float(limits.get("max_yaw_std", 0.20))
        )
        checks.append(
            _result(
                "AMCL /amcl_pose",
                healthy,
                "position_std=%.3fm yaw_std=%.3frad" % (position_std, yaw_std),
            )
        )

    listener = tf.TransformListener()
    try:
        listener.waitForTransform(
            "map", "base_link", rospy.Time(0), rospy.Duration(args.timeout)
        )
        translation, rotation = listener.lookupTransform(
            "map", "base_link", rospy.Time(0)
        )
        checks.append(
            _result(
                "TF map->base_link",
                True,
                "xyz=(%.3f, %.3f, %.3f) qz=%.3f qw=%.3f"
                % (
                    translation[0],
                    translation[1],
                    translation[2],
                    rotation[2],
                    rotation[3],
                ),
            )
        )
    except (tf.Exception, tf.LookupException, tf.ConnectivityException) as exc:
        checks.append(_result("TF map->base_link", False, str(exc)))

    move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    available = move_base.wait_for_server(rospy.Duration(args.timeout))
    checks.append(_result("move_base action", available, "server available" if available else "timeout"))

    image_topic = config.get("vision", {}).get("image_topic", "/ep_cam/image_raw")
    image, image_error = _wait_message(image_topic, Image, args.timeout)
    camera_required = args.require_camera
    checks.append(
        _result(
            "camera " + image_topic,
            image is not None or not camera_required,
            "fresh message" if image is not None else "optional: " + image_error,
        )
    )

    if not args.skip_calibration:
        metadata = config.get("_calibration", {})
        required = [
            ("parking", None),
            ("pickup", None),
            ("loading", None),
            ("rescue", 1),
            ("rescue", 2),
            ("rescue", 3),
            ("rescue", 4),
        ]
        for zone_name, zone_id in required:
            label = zone_name if zone_id is None else "%s:%d" % (zone_name, zone_id)
            checks.append(
                _result(
                    "calibration " + label,
                    calibration_has_zone(metadata, zone_name, zone_id),
                    metadata.get("path") or "calibration file missing",
                )
            )

    passed = all(checks)
    print("\nPREFLIGHT_%s" % ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

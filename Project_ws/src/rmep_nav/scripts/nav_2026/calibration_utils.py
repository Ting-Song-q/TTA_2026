#!/usr/bin/python3
# coding=UTF-8
"""Shared helpers for map-zone calibration and AMCL initial pose."""

import math
import time
from pathlib import Path

from zone_health import covariance_std

DEFAULT_INITIAL_COVARIANCE = [
    0.25, 0, 0, 0, 0, 0,
    0, 0.25, 0, 0, 0, 0,
    0, 0, 0.25, 0, 0, 0,
    0, 0, 0, 0.068, 0, 0,
    0, 0, 0, 0, 0.068, 0,
    0, 0, 0, 0, 0, 0.068,
]


def calibration_path(config, script_dir):
    filename = config.get("zone_entry", {}).get(
        "calibration_file", "zone_calibration.yaml"
    )
    path = Path(filename)
    if not path.is_absolute():
        path = Path(script_dir) / path
    return path


def mean_pose(samples):
    x = sum(sample[0] for sample in samples) / len(samples)
    y = sum(sample[1] for sample in samples) / len(samples)
    sin_yaw = sum(math.sin(sample[2]) for sample in samples)
    cos_yaw = sum(math.cos(sample[2]) for sample in samples)
    return x, y, math.atan2(sin_yaw, cos_yaw)


def max_position_spread(samples, center):
    return max(math.hypot(x - center[0], y - center[1]) for x, y, _ in samples)


def pose_dict(x, y, yaw):
    return {"x": float(x), "y": float(y), "yaw": float(yaw)}


def update_zone_overlay(
    overlay,
    zone_name,
    zone_id,
    pose,
    sync_bounds=True,
):
    key = int(zone_id) if zone_name == "rescue" else zone_name
    pose_data = pose_dict(pose[0], pose[1], pose[2])
    zones = overlay.setdefault("zones", {})
    if zone_name == "rescue":
        zones.setdefault("rescue", {})[key] = dict(pose_data)
    else:
        zones[key] = dict(pose_data)

    if sync_bounds:
        bounds = overlay.setdefault("zone_bounds", {})
        if zone_name == "rescue":
            entry = bounds.setdefault("rescue", {}).setdefault(key, {})
        else:
            entry = bounds.setdefault(key, {})
        entry["center"] = dict(pose_data)
    return pose_data


def _pose_from_mapping(pose):
    if not isinstance(pose, dict):
        return None
    return (
        float(pose.get("x", 0.0)),
        float(pose.get("y", 0.0)),
        float(pose.get("yaw", 0.0)),
    )


def resolve_initial_pose(config):
    loc = config.get("localization", {})
    if not loc.get("auto_initial_pose", False):
        return None

    explicit = loc.get("initial_pose")
    if isinstance(explicit, dict):
        pose = _pose_from_mapping(explicit)
        if pose is not None and (
            loc.get("allow_zero_pose", False)
            or abs(pose[0]) > 1e-6
            or abs(pose[1]) > 1e-6
        ):
            return pose

    source = loc.get("initial_pose_source", "parking")
    zones = config.get("zones", {})
    if source == "rescue":
        zone_id = loc.get("initial_pose_rescue_id", 1)
        rescue = zones.get("rescue", {})
        pose = _pose_from_mapping(rescue.get(zone_id) or rescue.get(str(zone_id)))
    else:
        pose = _pose_from_mapping(zones.get(source, {}))

    if pose is None:
        return None
    if not loc.get("allow_zero_pose", False) and abs(pose[0]) < 1e-6 and abs(pose[1]) < 1e-6:
        return None
    return pose


def zone_pose_from_config(config, zone_name, zone_id=None):
    zones = config.get("zones", {})
    if zone_name == "rescue":
        rescue = zones.get("rescue", {})
        pose = rescue.get(zone_id) or rescue.get(str(zone_id))
    else:
        pose = zones.get(zone_name)
    return _pose_from_mapping(pose)


def is_navigable_pose(pose, min_distance=0.05):
    if pose is None:
        return False
    return math.hypot(pose[0], pose[1]) >= min_distance


def collect_pose(listener, count, interval):
    import rospy
    from tf.transformations import euler_from_quaternion

    listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(5.0))
    samples = []
    for _ in range(count):
        translation, rotation = listener.lookupTransform(
            "map", "base_link", rospy.Time(0)
        )
        yaw = euler_from_quaternion(rotation)[2]
        samples.append((float(translation[0]), float(translation[1]), float(yaw)))
        rospy.sleep(interval)
    return samples


def check_localization_inputs(config, force=False):
    import rospy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from sensor_msgs.msg import LaserScan

    rospy.wait_for_message("/scan", LaserScan, timeout=2.0)
    amcl = rospy.wait_for_message(
        "/amcl_pose", PoseWithCovarianceStamped, timeout=2.0
    )
    position_std, yaw_std = covariance_std(amcl.pose.covariance)
    limits = config.get("zone_entry", {})
    max_position = float(limits.get("max_position_std", 0.15))
    max_yaw = float(limits.get("max_yaw_std", 0.20))
    if not force and (position_std > max_position or yaw_std > max_yaw):
        raise RuntimeError(
            "AMCL uncertainty too high: position_std=%.3f yaw_std=%.3f"
            % (position_std, yaw_std)
        )
    return position_std, yaw_std


def publish_initial_pose(x, y, yaw, z=0.0, duration=0.5, covariance=None):
    import rospy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from tf.transformations import quaternion_from_euler

    pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
    rospy.sleep(0.2)
    msg = PoseWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = float(z)
    q = quaternion_from_euler(0, 0, float(yaw))
    msg.pose.pose.orientation.x = q[0]
    msg.pose.pose.orientation.y = q[1]
    msg.pose.pose.orientation.z = q[2]
    msg.pose.pose.orientation.w = q[3]
    msg.pose.covariance = list(covariance or DEFAULT_INITIAL_COVARIANCE)
    deadline = time.time() + max(0.1, float(duration))
    while time.time() < deadline and not rospy.is_shutdown():
        pub.publish(msg)
        rospy.sleep(0.05)
    return msg


def wait_for_map_tf(listener, timeout=5.0):
    import rospy

    listener.waitForTransform(
        "map", "base_link", rospy.Time(0), rospy.Duration(timeout)
    )


def apply_auto_initial_pose(config, listener=None):
    import rospy
    import tf

    pose = resolve_initial_pose(config)
    if pose is None:
        return False

    loc = config.get("localization", {})
    publish_initial_pose(
        pose[0],
        pose[1],
        pose[2],
        duration=float(loc.get("publish_duration", 0.5)),
    )
    settle = float(loc.get("settle_time", 1.0))
    if settle > 0.0:
        rospy.sleep(settle)

    listener = listener or tf.TransformListener()
    try:
        wait_for_map_tf(listener, timeout=float(loc.get("tf_wait_timeout", 5.0)))
    except (
        tf.Exception,
        tf.LookupException,
        tf.ConnectivityException,
        tf.ExtrapolationException,
    ):
        if not loc.get("retry_relocate", False):
            rospy.logwarn("[Localization] auto initial pose published but TF not ready")
            return True
        rospy.logwarn("[Localization] TF not ready, republishing initial pose")
        publish_initial_pose(
            pose[0],
            pose[1],
            pose[2],
            duration=float(loc.get("publish_duration", 0.5)),
        )
        rospy.sleep(settle)
        wait_for_map_tf(listener, timeout=float(loc.get("tf_wait_timeout", 5.0)))

    rospy.loginfo(
        "[Localization] auto initial pose at (%.3f, %.3f, %.3f rad)",
        pose[0],
        pose[1],
        pose[2],
    )
    return True


def record_zone_pose(
    config,
    zone_name,
    zone_id=None,
    samples=10,
    interval=0.10,
    max_spread=0.03,
    force=False,
    sync_bounds=True,
    listener=None,
):
    import tf

    position_std, yaw_std = check_localization_inputs(config, force)
    listener = listener or tf.TransformListener()
    pose_samples = collect_pose(listener, samples, interval)
    pose = mean_pose(pose_samples)
    spread = max_position_spread(pose_samples, pose)
    if spread > max_spread and not force:
        raise RuntimeError(
            "pose was not stable: spread=%.3fm limit=%.3fm" % (spread, max_spread)
        )
    return {
        "pose": pose,
        "spread": spread,
        "position_std": position_std,
        "yaw_std": yaw_std,
    }

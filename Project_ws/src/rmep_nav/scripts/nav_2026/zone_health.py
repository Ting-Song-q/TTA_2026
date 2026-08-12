#!/usr/bin/python3
# coding=UTF-8

# BEGIN added: localization freshness and uncertainty checks
import math


def covariance_std(covariance):
    if covariance is None or len(covariance) < 36:
        return None, None
    position_variance = max(float(covariance[0]), float(covariance[7]), 0.0)
    yaw_variance = max(float(covariance[35]), 0.0)
    return math.sqrt(position_variance), math.sqrt(yaw_variance)


def evaluate_localization_health(metrics, config):
    config = config or {}
    checks = (
        ("scan_age", "scan_missing", "scan_stale", float(config.get("max_scan_age", 0.5))),
        ("tf_age", "tf_missing", "tf_stale", float(config.get("max_tf_age", 0.5))),
        (
            "amcl_pose_age",
            "amcl_pose_missing",
            "amcl_pose_stale",
            float(config.get("max_amcl_pose_age", 1.0)),
        ),
    )
    for key, missing_reason, stale_reason, limit in checks:
        value = metrics.get(key)
        if value is None:
            return False, missing_reason
        if value > limit:
            return False, "%s:%.3f>%.3f" % (stale_reason, value, limit)

    position_std = metrics.get("position_std")
    yaw_std = metrics.get("yaw_std")
    if position_std is None or yaw_std is None:
        return False, "amcl_covariance_missing"
    max_position_std = float(config.get("max_position_std", 0.15))
    max_yaw_std = float(config.get("max_yaw_std", 0.20))
    if position_std > max_position_std:
        return False, "position_uncertain:%.3f>%.3f" % (
            position_std,
            max_position_std,
        )
    if yaw_std > max_yaw_std:
        return False, "yaw_uncertain:%.3f>%.3f" % (yaw_std, max_yaw_std)
    return True, "localization_healthy"
# END added: localization freshness and uncertainty checks

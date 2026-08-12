#!/usr/bin/python3
# coding=UTF-8
"""激光反应式绕障：按「车体净空」判障并侧移绕行。

雷达读数是激光原点到障碍的距离，需减去雷达到车体外廓的偏置，
才得到真正可用于避障的车体净空：

    body_clearance = max(0, laser_range - lidar_to_body[side])

实测偏置（默认）：
    前 0.12 m / 后 0.31 m / 左·右 0.25 m

配置中的 safe_distance / critical_distance / emergency_stop_distance
均表示「车体外缘之外」的余量，而不是原始激光距离。
"""

from __future__ import print_function

import math

from geometry_msgs.msg import Twist

# 激光安装方式对应的扇区角度（度，相对 laser 帧约定）
# rear：0=车尾，±90=左右，180=车头（与 run_car / TF yaw=-pi 一致）
_MOUNT_SECTORS = {
    "front": {"front": 0, "back": 180, "left": 90, "right": -90},
    "rear": {"front": 180, "back": 0, "left": -90, "right": 90},
}

# 雷达到车体外廓距离 (m) —— 用户实测
DEFAULT_LIDAR_TO_BODY = {
    "front": 0.12,
    "back": 0.31,
    "left": 0.25,
    "right": 0.25,
}

# 车体净空阈值默认值 (m，车体外缘之外)
DEFAULT_BODY_THRESHOLDS = {
    "safe_distance": 0.20,
    "side_safe_distance": 0.15,
    "critical_distance": 0.10,
    "emergency_stop_distance": 0.05,
}


def sector_angles(cfg):
    """从配置解析前/后/左/右扇区中心角。"""
    custom = cfg.get("sectors")
    if custom:
        return {
            "front": custom.get("front", 0),
            "back": custom.get("back", 180),
            "left": custom.get("left", 90),
            "right": custom.get("right", -90),
        }
    mount = cfg.get("lidar_mount", "rear")
    return _MOUNT_SECTORS.get(mount, _MOUNT_SECTORS["rear"])


def lidar_to_body_offsets(cfg):
    """雷达原点到车体四边外廓的距离。"""
    offsets = dict(DEFAULT_LIDAR_TO_BODY)
    custom = cfg.get("lidar_to_body") or {}
    for key in offsets:
        if key in custom:
            offsets[key] = float(custom[key])
    # 兼容扁平字段
    for key, flat in (
        ("front", "lidar_to_front"),
        ("back", "lidar_to_back"),
        ("left", "lidar_to_left"),
        ("right", "lidar_to_right"),
    ):
        if flat in cfg:
            offsets[key] = float(cfg[flat])
    return offsets


def _angle_in_sector(angle_rad, center_deg, half_width_deg):
    """判断激光点是否落在扇区内（正确处理 ±π 环绕）。"""
    center = math.radians(center_deg)
    half = math.radians(half_width_deg)
    diff = math.atan2(
        math.sin(angle_rad - center), math.cos(angle_rad - center)
    )
    return abs(diff) <= half


def sector_min_distance(laser_data, center_deg, half_width_deg):
    """扇区内原始激光最近距离（未扣车体偏置）。"""
    if laser_data is None:
        return float("inf")

    best = float("inf")
    for i, r in enumerate(laser_data.ranges):
        ang = laser_data.angle_min + i * laser_data.angle_increment
        if not _angle_in_sector(ang, center_deg, half_width_deg):
            continue
        if not math.isfinite(r):
            continue
        if r < laser_data.range_min or r > laser_data.range_max:
            continue
        best = min(best, r)
    return best


def to_body_clearance(laser_range, offset, cfg=None):
    """
    原始激光距离 → 车体净空。

    明显落在车体内部的回波（自击）视为无效，返回 inf，避免误触发。
    """
    cfg = cfg or {}
    if not math.isfinite(laser_range):
        return float("inf")
    self_hit_margin = float(cfg.get("self_hit_margin", 0.03))
    if laser_range < max(0.0, offset - self_hit_margin):
        return float("inf")
    return max(0.0, laser_range - offset)


def get_raw_clearances(laser_data, cfg):
    """返回前/后/左/右扇区原始激光最近距离 (m)。"""
    half_w = cfg.get("sector_half_width", 25)
    side_half = cfg.get("side_sector_half_width", 55)
    sectors = sector_angles(cfg)
    inf = float("inf")
    if laser_data is None:
        return {"front": inf, "back": inf, "left": inf, "right": inf}
    return {
        "front": sector_min_distance(laser_data, sectors["front"], half_w),
        "back": sector_min_distance(laser_data, sectors["back"], half_w),
        "left": sector_min_distance(laser_data, sectors["left"], side_half),
        "right": sector_min_distance(laser_data, sectors["right"], side_half),
    }


def get_clearances(laser_data, cfg):
    """返回前/后/左/右「车体净空」最近障碍距离 (m)。"""
    raw = get_raw_clearances(laser_data, cfg)
    offsets = lidar_to_body_offsets(cfg)
    return {
        side: to_body_clearance(raw[side], offsets[side], cfg)
        for side in ("front", "back", "left", "right")
    }


def _clamp(value, limit):
    return max(-limit, min(limit, value))


def _threshold(cfg, key):
    if key in cfg:
        return float(cfg[key])
    return float(DEFAULT_BODY_THRESHOLDS.get(key, 0.0))


def compute_bypass_twist(laser_data, vx, vy, wz, cfg):
    """
    根据车体净空修正速度指令以实现绕障。
    vx/vy/wz 为期望速度（带方向符号）。
    """
    twist = Twist()
    twist.linear.x = vx
    twist.linear.y = vy
    twist.angular.z = wz

    if laser_data is None:
        return twist, "no_laser"

    safe = _threshold(cfg, "safe_distance")
    critical = _threshold(cfg, "critical_distance")
    bypass_speed = cfg.get("bypass_speed", 0.08)
    creep = cfg.get("creep_ratio", 0.25)
    side_safe = _threshold(cfg, "side_safe_distance")

    clearances = get_clearances(laser_data, cfg)
    front = clearances["front"]
    back = clearances["back"]
    left = clearances["left"]
    right = clearances["right"]

    mode = "clear"

    if vx > 0.01 and front < safe:
        mode = "bypass_forward"
        open_left = left > right
        twist.linear.y = bypass_speed if open_left else -bypass_speed
        twist.linear.x = vx * creep if front > critical else -bypass_speed * 0.5
        if front < critical and left < critical and right < critical:
            twist.linear.x = -bypass_speed
            mode = "bypass_forward_boxed"

    elif vx < -0.01 and back < safe:
        mode = "bypass_backward"
        open_left = left > right
        twist.linear.y = bypass_speed if open_left else -bypass_speed
        twist.linear.x = vx * creep if back > critical else bypass_speed * 0.5

    elif vy > 0.01 and left < side_safe:
        mode = "bypass_strafe_left"
        if front > back:
            twist.linear.x = bypass_speed * creep
        else:
            twist.linear.x = -bypass_speed * creep
        twist.linear.y = vy * creep if left > critical else bypass_speed * 0.5

    elif vy < -0.01 and right < side_safe:
        mode = "bypass_strafe_right"
        if front > back:
            twist.linear.x = bypass_speed * creep
        else:
            twist.linear.x = -bypass_speed * creep
        twist.linear.y = vy * creep if right > critical else -bypass_speed * 0.5

    elif abs(wz) > 0.01:
        if wz > 0 and right < side_safe:
            mode = "bypass_turn_left"
            twist.linear.y = bypass_speed
            twist.angular.z = wz * creep
        elif wz < 0 and left < side_safe:
            mode = "bypass_turn_right"
            twist.linear.y = -bypass_speed
            twist.angular.z = wz * creep

    max_v = cfg.get("max_linear_speed", 0.15)
    max_w = cfg.get("max_angular_speed", 0.4)
    twist.linear.x = _clamp(twist.linear.x, max_v)
    twist.linear.y = _clamp(twist.linear.y, max_v)
    twist.angular.z = _clamp(twist.angular.z, max_w)
    return twist, mode


def is_path_clear(laser_data, vx, vy, cfg):
    """判断主运动方向车体净空是否畅通，用于累计有效里程。"""
    if laser_data is None:
        return True

    safe = _threshold(cfg, "safe_distance")
    side_safe = _threshold(cfg, "side_safe_distance")
    clearances = get_clearances(laser_data, cfg)

    if vx > 0.01:
        return clearances["front"] > safe
    if vx < -0.01:
        return clearances["back"] > safe
    if vy > 0.01:
        return clearances["left"] > side_safe
    if vy < -0.01:
        return clearances["right"] > side_safe
    return True


def _relevant_clearances(clearances, vx=0.0, vy=0.0, wz=0.0, default_front=False):
    """按运动方向选取需要关注的扇区。"""
    checks = []
    if vx > 0.01:
        checks.append(clearances["front"])
    elif vx < -0.01:
        checks.append(clearances["back"])
    if vy > 0.01:
        checks.append(clearances["left"])
    elif vy < -0.01:
        checks.append(clearances["right"])
    if abs(wz) > 0.01:
        checks.append(clearances["right"] if wz > 0 else clearances["left"])
    if not checks and default_front:
        checks.append(clearances["front"])
    return checks


def is_emergency(laser_data, cfg, vx=0.0, vy=0.0, wz=0.0):
    """
    运动方向上的紧急停车：仅检查与当前速度指令相关的扇区净空。
    避免身后/侧面有墙时误触发全向急停。
    """
    if laser_data is None:
        return bool(cfg.get("fail_closed", False)) and (
            abs(vx) > 1e-4 or abs(vy) > 1e-4 or abs(wz) > 1e-4
        )
    emergency = _threshold(cfg, "emergency_stop_distance")
    clearances = get_clearances(laser_data, cfg)
    checks = _relevant_clearances(clearances, vx, vy, wz, default_front=False)
    if not checks:
        return False
    return min(checks) < emergency


def is_nav_emergency(laser_data, cfg):
    """move_base 导航监护：只关心车头前方净空是否即将碰撞。"""
    if laser_data is None:
        return False
    emergency = _threshold(cfg, "emergency_stop_distance")
    clearances = get_clearances(laser_data, cfg)
    return clearances["front"] < emergency


def is_move_base_blocked(laser_data, cfg):
    """
    move_base 前进路径被挡。
    默认仅在车头前方净空低于 critical_distance 时介入。
    move_base_guard_mode: critical | safe
    """
    if laser_data is None:
        return False

    clearances = get_clearances(laser_data, cfg)
    front = clearances["front"]
    mode = cfg.get("move_base_guard_mode", "critical")
    critical = _threshold(cfg, "critical_distance")

    if mode == "critical":
        return front < critical

    safe = _threshold(cfg, "safe_distance")
    if front < safe:
        return True
    side_safe = _threshold(cfg, "side_safe_distance")
    if front < safe * 1.2 and (
        clearances["left"] < side_safe or clearances["right"] < side_safe
    ):
        return True
    return False


def pick_bypass_direction(laser_data, cfg):
    """
    选择局部绕障横移方向 (+y 左 / -y 右)。
    返回 (vy, side_label)；两侧都不足时尝试后退。
    """
    clearances = get_clearances(laser_data, cfg)
    side_safe = _threshold(cfg, "side_safe_distance")
    bypass_speed = cfg.get("bypass_speed", 0.08)

    left_ok = clearances["left"] >= side_safe
    right_ok = clearances["right"] >= side_safe

    if left_ok and right_ok:
        open_left = clearances["left"] >= clearances["right"]
    elif left_ok:
        open_left = True
    elif right_ok:
        open_left = False
    else:
        return -bypass_speed * 0.5, "backward"

    vy = bypass_speed if open_left else -bypass_speed
    label = "left" if open_left else "right"
    return vy, label


def guard_twist(laser_data, vx, vy, wz, cfg):
    """对任意速度指令做激光修正，用于视觉对齐等细调场景。"""
    if laser_data is None or not cfg.get("enabled", False):
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = wz
        return twist
    twist, _ = compute_bypass_twist(laser_data, vx, vy, wz, cfg)
    return twist


def check_proximity_alert(laser_data, cfg):
    """
    检测贴近障碍（潜在碰撞），返回 (alert, min_dist, sector)。
    距离为车体净空。
    """
    if laser_data is None:
        return False, float("inf"), ""
    alert_dist = cfg.get(
        "collision_alert_distance", _threshold(cfg, "critical_distance")
    )
    clearances = get_clearances(laser_data, cfg)
    sector = min(clearances, key=clearances.get)
    min_dist = clearances[sector]
    return min_dist < alert_dist, min_dist, sector


def laser_threshold_for_body(side, body_clearance, cfg=None):
    """调试辅助：给定车体净空，换算成对应方向的原始激光阈值。"""
    offsets = lidar_to_body_offsets(cfg or {})
    return offsets[side] + float(body_clearance)

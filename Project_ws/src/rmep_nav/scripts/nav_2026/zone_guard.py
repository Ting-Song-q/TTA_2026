#!/usr/bin/python3
# coding=UTF-8

# BEGIN added: wheel-in-zone geometry module
"""Geometry helpers for zone-entry validation."""

import math


def _as_xyyaw(center):
    if center is None:
        return None
    if isinstance(center, dict):
        return (
            float(center.get("x", 0.0)),
            float(center.get("y", 0.0)),
            float(center.get("yaw", 0.0)),
        )
    if isinstance(center, (list, tuple)):
        if len(center) >= 3:
            return float(center[0]), float(center[1]), float(center[2])
        if len(center) == 2:
            return float(center[0]), float(center[1]), 0.0
    return None


def wheel_outer_points_base(vehicle_cfg):
    """Return wheel outer reference points in base_link."""
    vehicle_cfg = vehicle_cfg or {}
    size = vehicle_cfg.get("wheel_outer_size", {})
    length = float(size.get("length", 0.30))
    width = float(size.get("width", 0.24))
    off_x, off_y = vehicle_cfg.get("footprint_center_offset", [0.0, 0.0])
    off_x = float(off_x)
    off_y = float(off_y)

    hx = length / 2.0
    hy = width / 2.0
    return {
        "front_left": (off_x + hx, off_y + hy),
        "front_right": (off_x + hx, off_y - hy),
        "rear_left": (off_x - hx, off_y + hy),
        "rear_right": (off_x - hx, off_y - hy),
    }


def rectangle_polygon(center, size):
    """Build a rotated rectangle polygon from center pose and size."""
    pose = _as_xyyaw(center)
    if pose is None:
        return None
    x, y, yaw = pose
    if isinstance(size, dict):
        width = float(size.get("width", size.get("x", 0.8)))
        height = float(size.get("height", size.get("y", 0.8)))
    else:
        width = float(size[0])
        height = float(size[1])

    hx = width / 2.0
    hy = height / 2.0
    local = [
        (+hx, +hy),
        (+hx, -hy),
        (-hx, -hy),
        (-hx, +hy),
    ]
    c = math.cos(yaw)
    s = math.sin(yaw)
    polygon = []
    for px, py in local:
        mx = x + c * px - s * py
        my = y + s * px + c * py
        polygon.append((mx, my))
    return polygon


def transform_points(x, y, yaw, points):
    """Transform base_link points to map frame."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    out = {}
    for name, (px, py) in points.items():
        mx = x + c * px - s * py
        my = y + s * px + c * py
        out[name] = (mx, my)
    return out


def _normalize_polygon(polygon):
    if polygon is None:
        return []
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def _point_on_segment(point, a, b, eps=1e-9):
    px, py = point
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= eps


def _distance_point_to_segment(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / float(dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_in_polygon(point, polygon, margin=0.0):
    """Ray-casting point-in-polygon test with optional inward margin."""
    pts = _normalize_polygon(polygon)
    if len(pts) < 3:
        return False

    inside = False
    x, y = float(point[0]), float(point[1])
    n = len(pts)
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        if _point_on_segment((x, y), a, b):
            inside = True
            break
        ax, ay = a
        bx, by = b
        denom = by - ay
        if denom == 0:
            continue
        intersects = ((ay > y) != (by > y)) and (
            x < (bx - ax) * (y - ay) / denom + ax
        )
        if intersects:
            inside = not inside

    if not inside:
        return False

    if margin <= 0.0:
        return True

    min_dist = min(
        _distance_point_to_segment((x, y), pts[i], pts[(i + 1) % n])
        for i in range(n)
    )
    return min_dist >= margin


def count_points_in_polygon(points, polygon, margin=0.0):
    """Count points inside polygon."""
    if isinstance(points, dict):
        iterable = points.values()
    else:
        iterable = points
    count = 0
    for point in iterable:
        if point_in_polygon(point, polygon, margin=margin):
            count += 1
    return count


def polygon_center(polygon):
    """Return the arithmetic center of a valid polygon."""
    points = _normalize_polygon(polygon)
    if len(points) < 3:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def map_vector_to_base(dx, dy, yaw):
    """Rotate a map-frame displacement into base_link coordinates."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def clamp_vector(vector, max_length):
    """Limit a 2-D vector while preserving its direction."""
    x, y = float(vector[0]), float(vector[1])
    length = math.hypot(x, y)
    limit = max(0.0, float(max_length))
    if length == 0.0 or length <= limit:
        return x, y
    scale = limit / length
    return x * scale, y * scale


def angle_difference(target, current):
    """Return the shortest signed target-current angle in radians."""
    return math.atan2(math.sin(target - current), math.cos(target - current))


def safe_pose_error(current_pose, target_pose):
    """Return position, yaw and base-frame correction to a safe target pose."""
    current = _as_xyyaw(current_pose)
    target = _as_xyyaw(target_pose)
    if current is None or target is None:
        return None
    x, y, yaw = current
    target_x, target_y, target_yaw = target
    dx = target_x - x
    dy = target_y - y
    return {
        "position_error": math.hypot(dx, dy),
        "yaw_error": angle_difference(target_yaw, yaw),
        "correction_base": map_vector_to_base(dx, dy, yaw),
    }


def resolve_zone_polygon(
    zone_bounds_cfg,
    zone_name,
    zone_id=None,
    fallback_pose=None,
    default_size=(0.8, 0.8),
):
    """Resolve zone polygon from config or fallback pose."""
    zone_bounds_cfg = zone_bounds_cfg or {}
    entry = None
    if zone_name == "rescue":
        rescue_cfg = zone_bounds_cfg.get("rescue", {})
        if isinstance(rescue_cfg, dict) and zone_id is not None:
            entry = rescue_cfg.get(zone_id, rescue_cfg.get(str(zone_id)))
    else:
        entry = zone_bounds_cfg.get(zone_name)

    if entry:
        mode = entry.get("mode", "center_size")
        if mode == "polygon":
            return _normalize_polygon(entry.get("polygon", []))
        center = entry.get("center", fallback_pose)
        size = entry.get("size", default_size)
        return rectangle_polygon(center, size)

    if fallback_pose is None:
        return None
    return rectangle_polygon(fallback_pose, default_size)
# END added: wheel-in-zone geometry module

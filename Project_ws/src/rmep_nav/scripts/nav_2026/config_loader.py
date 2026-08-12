#!/usr/bin/python3
# coding=UTF-8

# BEGIN added: mission configuration and calibration overlay loader
from copy import deepcopy
from pathlib import Path

import yaml


def deep_merge(base, overlay):
    result = deepcopy(base or {})
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


def _mapping_get(node, key, default=None):
    """Read dict key; PyYAML 1.1 may turn bare key y/n into True/False."""
    if not isinstance(node, dict):
        return default
    if key in node:
        return node[key]
    if key == "y" and True in node:
        return node[True]
    if key == "n" and False in node:
        return node[False]
    return default


def as_pose(node):
    """Normalize a pose mapping to {x, y, yaw}; accept center-wrapped forms."""
    if not isinstance(node, dict):
        return {"x": 0.0, "y": 0.0, "yaw": 0.0}
    # Prefer top-level pose keys; unwrap center only when x/y are absent.
    # (deep_merge(zones, zone_bounds) may leave both pose keys and center.)
    if _mapping_get(node, "x", None) is None and _mapping_get(node, "y", None) is None:
        if isinstance(node.get("center"), dict):
            node = node["center"]
    return {
        "x": float(_mapping_get(node, "x", 0.0) or 0.0),
        "y": float(_mapping_get(node, "y", 0.0) or 0.0),
        "yaw": float(_mapping_get(node, "yaw", 0.0) or 0.0),
    }


def _normalize_zones(zones):
    if not isinstance(zones, dict):
        return {}
    out = {}
    for name, pose in zones.items():
        if name == "rescue" and isinstance(pose, dict):
            out["rescue"] = {key: as_pose(value) for key, value in pose.items()}
        else:
            out[name] = as_pose(pose)
    return out


def _sync_zone_bounds_centers(config):
    """用 zones 航点覆盖 zone_bounds.center，避免白名单与开环航点不一致。"""
    zones = _normalize_zones(config.get("zones") or {})
    config["zones"] = zones
    bounds = config.get("zone_bounds")
    if not isinstance(bounds, dict):
        return config

    def sync_one(bound_node, pose):
        if not isinstance(bound_node, dict) or not isinstance(pose, dict):
            return
        center = as_pose(bound_node.get("center") or pose)
        center["x"] = float(pose.get("x", center.get("x", 0.0)))
        center["y"] = float(pose.get("y", center.get("y", 0.0)))
        center["yaw"] = float(pose.get("yaw", center.get("yaw", 0.0)))
        bound_node["center"] = center

    for name in ("parking", "pickup", "loading"):
        if name in bounds and name in zones:
            sync_one(bounds[name], zones[name])

    rescue_bounds = bounds.get("rescue")
    rescue_zones = zones.get("rescue")
    if isinstance(rescue_bounds, dict) and isinstance(rescue_zones, dict):
        for key, pose in rescue_zones.items():
            node = rescue_bounds.get(key)
            if node is None:
                node = rescue_bounds.get(str(key))
            if node is not None:
                sync_one(node, pose)

    config["zone_bounds"] = bounds
    return config


def load_mission_config(path):
    path = Path(path).resolve()
    config = load_yaml(path)
    zone_entry = config.get("zone_entry", {}) or {}
    calibration_name = zone_entry.get("calibration_file")
    # 顶层 use_calibration 优先；默认 false：只用 mission_config.zones
    use_calibration = _as_bool(
        config.get("use_calibration", zone_entry.get("use_calibration", False)),
        False,
    )

    if not calibration_name:
        config["_calibration"] = {
            "loaded": False,
            "path": None,
            "zones": {},
            "applied": False,
        }
        return _sync_zone_bounds_centers(config)

    calibration_path = Path(calibration_name)
    if not calibration_path.is_absolute():
        calibration_path = path.parent / calibration_path
    overlay = load_yaml(calibration_path)
    if isinstance(overlay.get("zones"), dict):
        overlay["zones"] = _normalize_zones(overlay["zones"])
    calibrated_zones = _normalize_zones(
        deep_merge(overlay.get("zones", {}), overlay.get("zone_bounds", {}))
    )

    # 默认不合并标定；只有 use_calibration=true 才覆盖 zones / zone_bounds
    if use_calibration and overlay:
        config = deep_merge(config, overlay)
        if isinstance(config.get("zones"), dict):
            config["zones"] = _normalize_zones(config["zones"])
        config["_calibration"] = {
            "loaded": True,
            "path": str(calibration_path),
            "zones": calibrated_zones,
            "applied": True,
        }
    else:
        config["_calibration"] = {
            "loaded": bool(overlay),
            "path": str(calibration_path),
            "zones": calibrated_zones,
            "applied": False,
        }

    return _sync_zone_bounds_centers(config)


def calibration_has_zone(metadata, zone_name, zone_id=None):
    """Return whether the runtime overlay explicitly calibrated a zone."""
    metadata = metadata or {}
    if not metadata.get("loaded"):
        return False
    zones = metadata.get("zones", {})
    if zone_name == "rescue":
        rescue = zones.get("rescue", {})
        return zone_id in rescue or str(zone_id) in rescue
    return zone_name in zones


def save_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
    return path
# END added: mission configuration and calibration overlay loader

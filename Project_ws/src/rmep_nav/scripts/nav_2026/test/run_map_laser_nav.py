#!/usr/bin/python3
# coding=UTF-8
"""
基于雷达地图的小车运动 + 避障实机测试入口。

依赖栈（先启动）:
  1) roscore
  2) roslaunch rmep_base rmep_base.launch
  3) roslaunch rmep_nav map_amcl_move.launch   # /map + AMCL（可不启 move_base 也行）
  4) RViz 设 2D Pose Estimate（给 AMCL 初值）
  5) python3 run_map_laser_nav.py --rescue 2

航线默认: parking → pickup → loading → rescue_N → parking

模块说明见本文件底部 MODULES，或 --list-modules。
"""

from __future__ import print_function

import argparse
import math
import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
_NAV_DIR = _TEST_DIR.parent
if str(_NAV_DIR) not in sys.path:
    sys.path.insert(0, str(_NAV_DIR))
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from config_loader import as_pose, load_mission_config  # noqa: E402
from map_laser_nav import MapLaserNav  # noqa: E402

MODULES = [
    ("test/map_occupancy.py", "OccupancyMap", "订阅 /map，射线净空与线段占用检测"),
    ("test/map_laser_nav.py", "MapLaserNav", "AMCL 闭环追点 + TEB-style 多候选速度融合避障"),
    ("test/run_map_laser_nav.py", "本入口", "加载航点、跑 mission 航线"),
    ("laser_avoidance.py", "get_clearances/is_emergency/...", "激光扇区净空、急停、绕障方向"),
    ("openloop_duo.py", "LaserOpenLoopNav.face_yaw/turn_ang", "到位航向对齐（guard=False）"),
    ("config_loader.py", "load_mission_config/as_pose", "读取 mission_config 航点"),
    ("mission_config.yaml", "zones / obstacle_avoidance", "航点与避障阈值（可被本测试配置覆盖）"),
    ("mission_config_map_nav.yaml", "TEB-style params", "min_obstacle_dist/inflation_dist/weights 等参数"),
    ("rmep_base", "rmep_bringup + chassis_safety", "底盘 /cmd_vel、雷达 /scan"),
    ("rmep_nav launch", "map_server + amcl", "静态地图 /map、定位 /amcl_pose"),
    ("rplidar_ros", "rplidarNode", "激光驱动"),
]


def _print_modules():
    print("=== 本测试运用的模块 ===")
    for path, name, role in MODULES:
        print("  %-28s  %-36s  %s" % (path, name, role))


def _normalize_waypoints(zones):
    raw = dict(zones or {})
    waypoints = {}
    for name in ("parking", "pickup", "loading"):
        waypoints[name] = as_pose(raw.get(name))
    rescue = {}
    for key, pose in (raw.get("rescue") or {}).items():
        rescue[int(key)] = as_pose(pose)
    waypoints["rescue"] = rescue
    return waypoints


def load_config(config_path):
    import yaml

    path = Path(config_path)
    if not path.is_absolute():
        path = (_TEST_DIR / path).resolve()
    data = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    inherit = data.get("inherit_mission_config", "../mission_config.yaml")
    if inherit:
        mission_path = Path(inherit)
        if not mission_path.is_absolute():
            mission_path = (path.parent / mission_path).resolve()
        mission = load_mission_config(mission_path)
        waypoints = _normalize_waypoints(mission.get("zones"))
        avoid = dict(mission.get("obstacle_avoidance") or {})
        avoid.update(data.get("obstacle_avoidance") or {})
        map_nav = dict(data.get("map_nav") or {})
        car = dict(mission.get("car") or {})
        car.update(data.get("car") or {})
        return {
            "waypoints": waypoints,
            "obstacle_avoidance": avoid,
            "map_nav": map_nav,
            "car": car,
            "_config_path": str(path),
            "_mission_path": str(mission_path),
        }
    waypoints = _normalize_waypoints(data.get("zones") or data.get("waypoints"))
    return {
        "waypoints": waypoints,
        "obstacle_avoidance": dict(data.get("obstacle_avoidance") or {}),
        "map_nav": dict(data.get("map_nav") or {}),
        "car": dict(data.get("car") or {}),
        "_config_path": str(path),
        "_mission_path": None,
    }


def dump_plan(waypoints, rescue_zone):
    route = [
        ("parking", waypoints["parking"]),
        ("pickup", waypoints["pickup"]),
        ("loading", waypoints["loading"]),
        ("rescue_%d" % rescue_zone, waypoints["rescue"][rescue_zone]),
        ("parking", waypoints["parking"]),
    ]
    print("=== 计划航线（地图系）===")
    prev = None
    for name, pose in route:
        if prev is None:
            print(
                "  %s  (%.3f, %.3f, yaw=%.3f)"
                % (name, pose["x"], pose["y"], pose["yaw"])
            )
        else:
            dx = pose["x"] - prev["x"]
            dy = pose["y"] - prev["y"]
            print(
                "  %s  (%.3f, %.3f, yaw=%.3f)  |Δ|=%.3f"
                % (name, pose["x"], pose["y"], pose["yaw"], math.hypot(dx, dy))
            )
        prev = pose


def run_mission(nav, waypoints, rescue_zone):
    route = [
        ("pickup", waypoints["pickup"]),
        ("loading", waypoints["loading"]),
        ("rescue_%d" % rescue_zone, waypoints["rescue"][rescue_zone]),
        ("parking", waypoints["parking"]),
    ]
    for label, pose in route:
        if not nav.go_pose(pose, label=label):
            nav.stop()
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="基于雷达地图的小车运动与避障测试"
    )
    parser.add_argument(
        "--config",
        default="mission_config_map_nav.yaml",
        help="测试配置（默认继承 ../mission_config.yaml）",
    )
    parser.add_argument("--rescue", type=int, default=None, help="救援区 1-4")
    parser.add_argument(
        "--dump-plan", action="store_true", help="只打印航线，不运动"
    )
    parser.add_argument(
        "--list-modules", action="store_true", help="列出运用的全部模块"
    )
    parser.add_argument(
        "--single",
        choices=("pickup", "loading", "rescue", "parking"),
        default=None,
        help="只跑单个目标点",
    )
    args = parser.parse_args()

    if args.list_modules:
        _print_modules()
        return 0

    cfg = load_config(args.config)
    waypoints = cfg["waypoints"]
    car = cfg["car"]
    rescue_zone = int(
        args.rescue if args.rescue is not None else car.get("rescue_zone", 2)
    )
    if rescue_zone not in waypoints["rescue"]:
        print("无效救援区 %s，可选 %s" % (rescue_zone, sorted(waypoints["rescue"])))
        return 2

    print("[MapNavTest] config=%s" % cfg["_config_path"])
    print("[MapNavTest] mission=%s" % cfg["_mission_path"])
    dump_plan(waypoints, rescue_zone)
    _print_modules()

    if args.dump_plan:
        return 0

    nav = MapLaserNav(
        speed=float(car.get("speed", 0.25)),
        turn_speed=float(car.get("turn_speed", 0.45)),
        avoidance_cfg=cfg["obstacle_avoidance"],
        map_cfg=cfg["map_nav"],
        log_prefix="[MapNav]",
        node_name="run_map_laser_nav",
    )

    try:
        if args.single:
            if args.single == "rescue":
                pose = waypoints["rescue"][rescue_zone]
                label = "rescue_%d" % rescue_zone
            else:
                pose = waypoints[args.single]
                label = args.single
            ok = nav.go_pose(pose, label=label)
        else:
            ok = run_mission(nav, waypoints, rescue_zone)
    finally:
        nav.stop()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

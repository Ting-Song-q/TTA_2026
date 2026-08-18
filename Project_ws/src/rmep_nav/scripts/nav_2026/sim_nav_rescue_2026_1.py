#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在无 ROS 环境下离线模拟 nav_rescue_2026_1.py。

用法:
  python sim_nav_rescue_2026_1.py --skip-drone --rescue 2
  python sim_nav_rescue_2026_1.py --skip-drone --rescue 2 --obstacle
  python sim_nav_rescue_2026_1.py --skip-drone --rescue 2 --speed-up 100
"""

from __future__ import print_function

import argparse
import math
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


_HERE = Path(__file__).resolve().parent
_SPEED_UP = 50.0
_CMD_LOG = []  # type: List[dict]
_WALL0 = time.time()


def _sim_now():
    return (_WALL0 + (time.time() - _WALL0) * _SPEED_UP)


# ---------------------------------------------------------------------------
# Mock ROS / msgs / rescue_protocol（须在 import 目标脚本前注入）
# ---------------------------------------------------------------------------


class _Time(object):
    def __init__(self, secs=0.0):
        self._secs = float(secs)

    def to_sec(self):
        return self._secs

    @staticmethod
    def now():
        return _Time(_sim_now())


class _Rate(object):
    def __init__(self, hz):
        self._dt = 1.0 / max(1.0, float(hz))

    def sleep(self):
        time.sleep(self._dt / _SPEED_UP)


class _Publisher(object):
    def __init__(self, topic, msg_type, queue_size=10):
        self.topic = topic

    def publish(self, msg):
        vx = float(getattr(msg.linear, "x", 0.0))
        vy = float(getattr(msg.linear, "y", 0.0))
        wz = float(getattr(msg.angular, "z", 0.0))
        if abs(vx) < 1e-9 and abs(vy) < 1e-9 and abs(wz) < 1e-9:
            kind = "stop"
        else:
            kind = "cmd"
        _CMD_LOG.append(
            {
                "t": _sim_now() - _WALL0,
                "kind": kind,
                "vx": vx,
                "vy": vy,
                "wz": wz,
            }
        )


class _Subscriber(object):
    def __init__(self, topic, msg_type, callback, queue_size=1):
        self.topic = topic
        self.callback = callback
        # 立刻喂一帧，供 _decide_sidestep 使用
        if callable(callback):
            callback(_make_scan(_Subscriber.obstacle))


_Subscriber.obstacle = False


class _Rospy(types.ModuleType):
    def __init__(self):
        super(_Rospy, self).__init__("rospy")
        self.Time = _Time
        self.Rate = _Rate
        self.Publisher = _Publisher
        self.Subscriber = _Subscriber
        self._shutdown = False

    def init_node(self, *a, **k):
        print("[sim] rospy.init_node(%s)" % (a[0] if a else ""))

    def is_shutdown(self):
        return self._shutdown

    def sleep(self, duration):
        time.sleep(max(0.0, float(duration)) / _SPEED_UP)

    def spin(self):
        pass

    def loginfo(self, fmt, *args):
        print("[INFO] " + (fmt % args if args else str(fmt)))

    def logwarn(self, fmt, *args):
        print("[WARN] " + (fmt % args if args else str(fmt)))

    def logerr(self, fmt, *args):
        print("[ERR ] " + (fmt % args if args else str(fmt)))


class _Vec(object):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class Twist(object):
    def __init__(self):
        self.linear = _Vec()
        self.angular = _Vec()


class LaserScan(object):
    def __init__(self):
        self.angle_min = -math.pi
        self.angle_max = math.pi
        self.angle_increment = math.radians(1.0)
        self.range_min = 0.05
        self.range_max = 12.0
        self.ranges = []


def _make_scan(with_obstacle):
    """rear mount: 0°=车尾, 180°=车头。有障时在车头方向放近距点。"""
    scan = LaserScan()
    n = int(round((scan.angle_max - scan.angle_min) / scan.angle_increment)) + 1
    ranges = [8.0] * n
    if with_obstacle:
        # 车头 ≈ 180° → angle = π
        for i in range(n):
            ang = scan.angle_min + i * scan.angle_increment
            # 前向扇区（相对 rear：中心 π）
            diff = abs(math.atan2(math.sin(ang - math.pi), math.cos(ang - math.pi)))
            if diff <= math.radians(25):
                # 原始激光 0.50 → 净空 ≈ 0.50-0.12=0.38 < 0.6
                ranges[i] = 0.50
    scan.ranges = ranges
    return scan


def _install_mocks():
    rospy = _Rospy()
    sys.modules["rospy"] = rospy

    geo = types.ModuleType("geometry_msgs")
    geo_msg = types.ModuleType("geometry_msgs.msg")
    geo_msg.Twist = Twist
    geo.msg = geo_msg
    sys.modules["geometry_msgs"] = geo
    sys.modules["geometry_msgs.msg"] = geo_msg

    sensor = types.ModuleType("sensor_msgs")
    sensor_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msg.LaserScan = LaserScan
    sensor.msg = sensor_msg
    sys.modules["sensor_msgs"] = sensor
    sys.modules["sensor_msgs.msg"] = sensor_msg

    @dataclass
    class RescueOrder(object):
        zone: int
        level: int
        all_orders: List = field(default_factory=list, repr=False)

    rescue = types.ModuleType("rescue_protocol")
    rescue.RescueOrder = RescueOrder
    rescue.notify_loading_done = lambda **k: None
    rescue.notify_unload_done = lambda **k: None
    rescue.wait_for_delivery_done = lambda **k: True
    rescue.wait_for_rescue_target = lambda **k: RescueOrder(zone=2, level=1)
    sys.modules["rescue_protocol"] = rescue

    return rospy


def _summarize_cmds(log):
    """合并连续相同速度指令，统计有效运动段。"""
    segments = []
    i = 0
    while i < len(log):
        e = log[i]
        if e["kind"] == "stop":
            i += 1
            continue
        j = i
        while j < len(log) and log[j]["kind"] == "cmd":
            same = (
                abs(log[j]["vx"] - e["vx"]) < 1e-9
                and abs(log[j]["vy"] - e["vy"]) < 1e-9
                and abs(log[j]["wz"] - e["wz"]) < 1e-9
            )
            if not same:
                break
            j += 1
        t0 = e["t"]
        t1 = log[j - 1]["t"] if j > i else e["t"]
        # 粗算：控制周期约 1/100s 仿真时间
        dur = max(t1 - t0, 0.0) + 0.01
        segments.append(
            {
                "vx": e["vx"],
                "vy": e["vy"],
                "wz": e["wz"],
                "t0": t0,
                "duration≈": round(dur, 2),
            }
        )
        i = j
    return segments


def main():
    global _SPEED_UP, _WALL0, _CMD_LOG

    parser = argparse.ArgumentParser(description="离线模拟 nav_rescue_2026_1")
    parser.add_argument("--config", type=str, default=str(_HERE / "nav_rescue_2026_1.yaml"))
    parser.add_argument("--skip-drone", action="store_true", default=True)
    parser.add_argument("--rescue", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument(
        "--obstacle",
        action="store_true",
        help="模拟前方 0.6m 内有障碍，触发侧移绕障",
    )
    parser.add_argument("--speed-up", type=float, default=80.0, help="时间加速倍数")
    args = parser.parse_args()

    _SPEED_UP = max(1.0, float(args.speed_up))
    _CMD_LOG = []
    _WALL0 = time.time()
    _Subscriber.obstacle = bool(args.obstacle)

    _install_mocks()
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    # 延迟导入，确保 mocks 已就位
    import nav_rescue_2026_1 as nav  # noqa: E402

    print("=" * 60)
    print(
        "模拟 nav_rescue_2026_1 | rescue=%d | obstacle=%s | speed_up=%.0fx"
        % (args.rescue, args.obstacle, _SPEED_UP)
    )
    print("config:", args.config)
    print("=" * 60)

    wall_t0 = time.time()
    node = nav.NavTimed(
        config_path=args.config,
        skip_drone=True,
        rescue_zone=args.rescue,
        cmd_vel_topic="/cmd_vel",
        autostart=True,
    )
    wall_dt = time.time() - wall_t0
    sim_dt = (_sim_now() - _WALL0)

    segs = _summarize_cmds(_CMD_LOG)
    print()
    print("=" * 60)
    print("运动段摘要（合并后的 cmd_vel，仿真时间）")
    print("=" * 60)
    for idx, s in enumerate(segs, 1):
        print(
            "%2d) t=%.1fs  dur≈%.2fs  vx=%+.3f vy=%+.3f wz=%+.3f"
            % (idx, s["t0"], s["duration≈"], s["vx"], s["vy"], s["wz"])
        )

    print()
    print(
        "完成: need_sidestep=%s | 发布次数=%d | 运动段=%d | "
        "仿真时长≈%.1fs | 墙钟%.1fs"
        % (
            getattr(node, "_need_sidestep", None),
            len(_CMD_LOG),
            len(segs),
            sim_dt,
            wall_dt,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

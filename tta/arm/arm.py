#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 SO-101 各舵机当前 ticks 位姿。

用法:
  python3 arm.py
  python3 arm.py --loop 0.5
  python3 arm.py --yaml
  python3 arm.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from start import (  # noqa: E402
    DEFAULT_BAUD,
    MOTORS,
    connect_bus,
    enable_torque_safe,
    read_positions,
)

DEFAULT_CONFIG = _HERE / "og.yaml"
JOINTS = list(MOTORS.keys())


def load_port_baud(config: Path) -> tuple[str, int]:
    port = "/dev/ttyACM1"
    baud = DEFAULT_BAUD
    if config.exists():
        with config.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            port = str(data.get("port") or port)
            if data.get("baud") is not None:
                baud = int(data["baud"])
    return port, baud


def print_ticks(title: str, pose: Dict[str, float], *, as_yaml: bool = False) -> None:
    print(title)
    if as_yaml:
        print("  # 可粘贴到 og.yaml 的 poses.*")
        for name in JOINTS:
            if name in pose:
                print(f"  {name}: {int(round(float(pose[name])))}")
        return
    for name in JOINTS:
        if name in pose:
            print(f"  {name}: {int(round(float(pose[name])))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="读取 SO-101 各舵机 ticks")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port", default=None, help="串口路径（默认读 og.yaml）")
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument(
        "--loop",
        type=float,
        default=None,
        metavar="SEC",
        help="按间隔秒数连续读取（Ctrl+C 退出）",
    )
    parser.add_argument(
        "--yaml",
        action="store_true",
        help="按 og.yaml poses 格式打印，便于复制",
    )
    parser.add_argument(
        "--torque-on",
        action="store_true",
        help="读取前使能力矩（默认不改力矩状态）",
    )
    args = parser.parse_args()

    cfg_port, cfg_baud = load_port_baud(args.config)
    port = args.port or cfg_port
    baud = int(args.baud if args.baud is not None else cfg_baud)

    print(f"[arm] 连接 {port} @ {baud}")
    bus = connect_bus(port, baud=baud, configure=False)
    try:
        if args.torque_on:
            enable_torque_safe(bus)
            print("[torque] ON")
        else:
            print("[info] 未改力矩；仅读 Present_Position")

        def once() -> Dict[str, float]:
            pose = {k: float(v) for k, v in read_positions(bus).items()}
            print_ticks("当前舵机 ticks:", pose, as_yaml=bool(args.yaml))
            return pose

        if args.loop is None:
            once()
            return 0

        period = max(0.05, float(args.loop))
        print(f"[loop] 每 {period:.2f}s 读取一次，Ctrl+C 退出")
        while True:
            once()
            print("---")
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n用户中断")
        return 130
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass
        print("已断开串口")


if __name__ == "__main__":
    raise SystemExit(main())

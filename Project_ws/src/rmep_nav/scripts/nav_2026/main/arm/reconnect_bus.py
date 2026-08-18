#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取结束后重新握手机械臂总线（不改位姿、不松力矩）。

用于 mobile_grasp 断开串口后，在导航前往装货区前确认舵机在线。
若夹爪 id=6 过载掉线：默认降级只握手 1–5，避免整段任务卡死。

用法:
  python3 reconnect_bus.py --yes
  python3 reconnect_bus.py --yes --config reset_and_grasp.yaml
  python3 reconnect_bus.py --yes --max-attempts 4 --retry-s 1.0
  python3 reconnect_bus.py --yes --require-gripper
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from og import TicksArm  # noqa: E402
from start import DEFAULT_BAUD  # noqa: E402


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a YAML mapping: %s" % path)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取后重连机械臂舵机")
    parser.add_argument("--yes", action="store_true", help="允许打开串口")
    parser.add_argument(
        "--config",
        type=Path,
        default=_HERE / "reset_and_grasp.yaml",
        help="读取 port/baud/sequence 的 YAML",
    )
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--retry-s", type=float, default=None)
    parser.add_argument(
        "--hold-s",
        type=float,
        default=0.3,
        help="握手成功后保持连接再断开的秒数",
    )
    parser.add_argument(
        "--allow-missing-gripper",
        action="store_true",
        default=True,
        help="缺夹爪 id=6 时降级握手 1–5（默认开启）",
    )
    parser.add_argument(
        "--require-gripper",
        action="store_true",
        help="必须 1–6 全在线，不允许降级",
    )
    args = parser.parse_args()
    if not args.yes:
        print("Refusing: pass --yes")
        return 2

    allow_missing_gripper = bool(args.allow_missing_gripper) and not bool(
        args.require_gripper
    )

    cfg = load_cfg(args.config.resolve())
    sequence = dict(cfg.get("sequence") or {})
    motion = dict(cfg.get("motion") or {})
    port = str(args.port or cfg.get("port", "/dev/ttyACM1"))
    baud = int(args.baud or cfg.get("baud") or DEFAULT_BAUD)
    max_attempts = max(
        1,
        int(
            args.max_attempts
            if args.max_attempts is not None
            else sequence.get("connect_max_attempts", 4)
        ),
    )
    retry_s = max(
        0.0,
        float(
            args.retry_s
            if args.retry_s is not None
            else sequence.get("connect_retry_s", 1.0)
        ),
    )
    hold_s = max(0.0, float(args.hold_s))

    print(
        f"[reconnect] port={port} baud={baud} "
        f"attempts={max_attempts} retry_s={retry_s:.1f} "
        f"allow_missing_gripper={allow_missing_gripper}"
    )
    arm = TicksArm(port, baud, motion)
    try:
        try:
            arm.connect_with_retry(
                max_attempts=max_attempts,
                retry_s=retry_s,
                allow_missing_gripper=allow_missing_gripper,
            )
        except KeyboardInterrupt:
            print("\n[reconnect] 用户中断")
            return 130
        ticks = arm.read()
        gripper = ticks.get("gripper")
        if gripper is None:
            print(
                "[reconnect] OK arm 1–5 online；夹爪未在总线 "
                "(持块可继续导航；放置前请断电复位 id=6)"
            )
        else:
            print(f"[reconnect] OK motors online; gripper={gripper:.0f}")
        if hold_s > 0:
            time.sleep(hold_s)
        return 0
    except (ConnectionError, OSError, RuntimeError, TimeoutError) as exc:
        err = str(exc)
        if "Missing motor" in err or "id: model_number" in err:
            print(
                f"[reconnect] 舵机握手失败:\n{exc}\n"
                f"已重试 {max_attempts} 次。请断电复位机械臂后再继续。"
            )
        else:
            print(f"[reconnect] 连接失败: {exc}")
        return 1
    finally:
        # 不断力矩，保持抓取后的持块状态
        arm.disconnect(release_torque=False)
        print("[reconnect] 已断开串口；力矩未主动关闭")


if __name__ == "__main__":
    raise SystemExit(main())

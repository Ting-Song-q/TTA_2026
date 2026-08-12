#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂抓取动作脚本（LeRobot + Feetech SCServo）。

依赖（在 lerobot conda 环境中）:
  pip install "lerobot[feetech]"
  # 或: pip install feetech-servo-sdk pyserial

用法（Linux）:
  python3 start.py --port /dev/ttyACM0           # 执行抓取动作
  python3 start.py --port /dev/ttyACM0 --read    # 只读当前位置
  python3 start.py --list-ports
  python3 start.py --scan

注意:S
  - 抓取各关节 ticks 在下方 GRASP_* 中修改（现场标定）
  - 若夹爪开合方向反了，交换 OPEN_GRIPPER / CLOSE_GRIPPER
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 按你的机械臂修改这里（现场: /dev/ttyACM1 @ 1Mbps，ID 1..5 已确认）
# ---------------------------------------------------------------------------
# norm_mode:
#   RANGE_M100_100  关节常用（归一化约 -100~100）
#   RANGE_0_100     夹爪常用（归一化约 0~100）
#   DEGREES         需要标定后更接近真实角度
# 关节名仅便于调用；若实际关节顺序不同，只改 id 映射即可。
# SO-101 / Feetech 常见为 ID 1..6（不是 0..5）
MOTORS = {
    "shoulder_pan": {"id": 1, "model": "sts3215", "norm_mode": "RANGE_M100_100"},
    "shoulder_lift": {"id": 2, "model": "sts3215", "norm_mode": "RANGE_M100_100"},
    "elbow_flex": {"id": 3, "model": "sts3215", "norm_mode": "RANGE_M100_100"},
    "wrist_flex": {"id": 4, "model": "sts3215", "norm_mode": "RANGE_M100_100"},
    "wrist_roll": {"id": 5, "model": "sts3215", "norm_mode": "RANGE_M100_100"},
    "gripper": {"id": 6, "model": "sts3215", "norm_mode": "RANGE_0_100"},
}

# Feetech: Protocol 0 常见于 SCS / 部分 STS；Protocol 1 常见于较新 STS/SMS
PROTOCOL_VERSION = 0
# 现场命中: protocol=0 baud=1Mbps
DEFAULT_BAUD = 1_000_000
# 扫描时依次尝试（出厂多为 1Mbps；也有被改成 115200 的）
SCAN_BAUDS = (1_000_000, 500_000, 250_000, 115_200, 57_600)
SCAN_PROTOCOLS = (0, 1)
# 廉价 USB 半双工适配器上，写寄存器更容易丢包，需要重试和间隔
WRITE_RETRIES = 8
INTER_CMD_DELAY_S = 0.03
BUS_TIMEOUT_MS = 100

# ---------------------------------------------------------------------------
# 抓取动作位姿（原始 ticks，约 0~4095；中位常见约 2048）
# 流程: 观察位张开 → 接近 → 下降 → 闭合 → 抬起 → 收回
# 请按现场改数值；幅度先保守，确认方向正确后再加大。
# ---------------------------------------------------------------------------
OPEN_GRIPPER = 1200
CLOSE_GRIPPER = 2100
MOVE_SETTLE_S = 1.2
GRASP_SETTLE_S = 1.0

# 观察 / 待机（夹爪张开）
GRASP_OBSERVE: Dict[str, int] = {
    "shoulder_pan": 2048,
    "shoulder_lift": 1900,
    "elbow_flex": 2300,
    "wrist_flex": 2048,
    "wrist_roll": 2048,
    "gripper": OPEN_GRIPPER,
}
# 高位接近目标上方
GRASP_APPROACH: Dict[str, int] = {
    "shoulder_pan": 2048,
    "shoulder_lift": 1750,
    "elbow_flex": 2450,
    "wrist_flex": 2100,
    "wrist_roll": 2048,
    "gripper": OPEN_GRIPPER,
}
# 下降到抓取点（仍张开）
GRASP_DOWN: Dict[str, int] = {
    "shoulder_pan": 2048,
    "shoulder_lift": 1650,
    "elbow_flex": 2550,
    "wrist_flex": 2150,
    "wrist_roll": 2048,
    "gripper": OPEN_GRIPPER,
}
# 抓起后抬升（夹爪闭合）
GRASP_LIFT: Dict[str, int] = {
    "shoulder_pan": 2048,
    "shoulder_lift": 1900,
    "elbow_flex": 2300,
    "wrist_flex": 2048,
    "wrist_roll": 2048,
    "gripper": CLOSE_GRIPPER,
}
# 收回到观察位（保持闭合，方便确认抓住了）
GRASP_HOME: Dict[str, int] = {
    "shoulder_pan": 2048,
    "shoulder_lift": 1900,
    "elbow_flex": 2300,
    "wrist_flex": 2048,
    "wrist_roll": 2048,
    "gripper": CLOSE_GRIPPER,
}


def list_serial_ports() -> List[str]:
    """列出可用串口。"""
    try:
        from serial.tools import list_ports

        ports = [p.device for p in list_ports.comports()]
        if ports:
            return ports
    except Exception:
        pass

    # 兜底：按平台常见设备名扫描
    patterns = [
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
        "COM*",
    ]
    found: List[str] = []
    for pat in patterns:
        found.extend(glob.glob(pat))
    return sorted(set(found))


def describe_port(port: str) -> None:
    """打印串口对应的 USB 设备信息，便于判断是不是舵机适配器。"""
    try:
        from serial.tools import list_ports

        for p in list_ports.comports():
            if p.device != port:
                continue
            print(f"串口信息: {p.device}")
            print(f"  desc     : {p.description}")
            print(f"  hwid     : {p.hwid}")
            print(f"  manufacturer: {getattr(p, 'manufacturer', None)}")
            print(f"  product  : {getattr(p, 'product', None)}")
            return
    except Exception as exc:
        print(f"[warn] 无法读取串口信息: {exc}")
    print(f"串口信息: {port} （无详细 USB 描述）")


def auto_pick_port() -> Optional[str]:
    """优先选择 Linux USB 串口。"""
    ports = list_serial_ports()
    preferred = [p for p in ports if "ttyACM" in p or "ttyUSB" in p]
    if preferred:
        return preferred[0]
    return ports[0] if ports else None


def build_motors():
    try:
        from lerobot.motors import Motor, MotorNormMode
    except ImportError:
        from lerobot.motors.motors_bus import Motor, MotorNormMode

    motors = {}
    for name, cfg in MOTORS.items():
        norm = getattr(MotorNormMode, cfg.get("norm_mode", "RANGE_M100_100"))
        # 新版签名: Motor(id, model, norm_mode)
        motors[name] = Motor(cfg["id"], cfg["model"], norm)
    return motors


def _clear_port(bus) -> None:
    try:
        bus.port_handler.clearPort()
        bus.port_handler.is_using = False
    except Exception:
        pass


def safe_write(
    bus,
    data_name: str,
    motor: str,
    value: float,
    *,
    normalize: bool = False,
    retries: int = WRITE_RETRIES,
) -> None:
    """带重试的单寄存器写入；避开 LeRobot enable_torque 里额外的 Lock 写。"""
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            _clear_port(bus)
            bus.write(
                data_name,
                motor,
                value,
                normalize=normalize,
                num_retry=0,
            )
            time.sleep(INTER_CMD_DELAY_S)
            return
        except BaseException as exc:  # noqa: BLE001 - 通信层错误类型不一
            last = exc
            time.sleep(0.04 * (attempt + 1))
            _clear_port(bus)
    raise ConnectionError(
        f"写入失败 {data_name}@{motor}={value}（重试 {retries} 次）: {last}"
    ) from last


def enable_torque_safe(bus, motors: Optional[List[str]] = None) -> None:
    """只写 Torque_Enable=1，不写 Lock（Lock 在部分总线上会触发 Incorrect status packet）。"""
    names = list(MOTORS.keys()) if motors is None else motors
    for name in names:
        safe_write(bus, "Torque_Enable", name, 1, normalize=False)
        print(f"  torque ON : {name}")


def disable_torque_safe(bus, motors: Optional[List[str]] = None) -> None:
    names = list(MOTORS.keys()) if motors is None else motors
    for name in names:
        try:
            safe_write(bus, "Torque_Enable", name, 0, normalize=False)
        except Exception as exc:
            print(f"  [warn] torque OFF {name}: {exc}")


def connect_bus(port: str, baud: Optional[int] = None, configure: bool = False):
    """连接 Feetech 总线（LeRobot 封装）。

    LeRobot 默认按 1Mbps 握手；本臂实际为 115200，必须先设波特率再握手。
    """
    from lerobot.motors.feetech import FeetechMotorsBus

    rate = DEFAULT_BAUD if baud is None else baud
    bus = FeetechMotorsBus(
        port=port,
        motors=build_motors(),
        protocol_version=PROTOCOL_VERSION,
    )
    # 打开串口但不握手 → 切到正确波特率 → 再做电机存在性检查
    bus.connect(handshake=False)
    bus.set_baudrate(rate)
    try:
        bus.set_timeout(BUS_TIMEOUT_MS)
    except Exception:
        pass
    print(f"[OK] {port} @ baud={rate} protocol={PROTOCOL_VERSION}")
    try:
        bus._handshake()
    except Exception:
        bus.disconnect(disable_torque=False)
        raise

    # configure_motors 会写 Acceleration 等，在部分适配器/固件上经常失败，默认跳过
    if configure:
        try:
            bus.configure_motors()
        except Exception as exc:
            print(f"[warn] configure_motors 跳过: {exc}")
    return bus


def read_positions(bus) -> Dict[str, float]:
    """读取原始位置值（ticks）。未校准时必须 normalize=False。"""
    names = list(MOTORS.keys())
    # 优先逐个读：比 sync_read 更稳（半双工 USB）
    out: Dict[str, float] = {}
    for n in names:
        last: Optional[BaseException] = None
        for attempt in range(WRITE_RETRIES):
            try:
                _clear_port(bus)
                try:
                    out[n] = bus.read("Present_Position", n, normalize=False, num_retry=0)
                except TypeError:
                    out[n] = bus.read("Present_Position", n, normalize=False)
                time.sleep(INTER_CMD_DELAY_S)
                last = None
                break
            except BaseException as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.03 * (attempt + 1))
        if last is not None:
            raise ConnectionError(f"读取 Present_Position@{n} 失败: {last}") from last
    return out


def move_to(bus, goals: Dict[str, float], wait_s: float = 1.5) -> None:
    """逐个写 Goal_Position（原始 ticks）。比 sync_write 更适合不稳的半双工总线。"""
    for name, value in goals.items():
        safe_write(bus, "Goal_Position", name, int(value), normalize=False)
    time.sleep(wait_s)


def write_pose(bus, pose: Dict[str, int], settle: float, label: str = "") -> None:
    """移动到完整关节位姿并打印步骤。"""
    if label:
        print(f"[{label}]")
        for name in MOTORS:
            if name in pose:
                print(f"  {name}: {int(pose[name])}")
    move_to(bus, {k: int(v) for k, v in pose.items()}, wait_s=settle)


def set_gripper(bus, value: int, settle: float = 0.6, label: str = "") -> None:
    if label:
        print(f"[{label}] gripper -> {value}")
    move_to(bus, {"gripper": int(value)}, wait_s=settle)


def run_grasp_sequence(bus) -> None:
    """执行开环抓取动作序列。"""
    print("开始抓取动作序列...")
    enable_torque_safe(bus)
    time.sleep(0.2)

    write_pose(bus, GRASP_OBSERVE, MOVE_SETTLE_S, "1/6 观察位 + 张开夹爪")
    set_gripper(bus, OPEN_GRIPPER, 0.5)

    write_pose(bus, GRASP_APPROACH, MOVE_SETTLE_S, "2/6 接近目标上方")
    write_pose(bus, GRASP_DOWN, MOVE_SETTLE_S, "3/6 下降到抓取点")
    set_gripper(bus, CLOSE_GRIPPER, GRASP_SETTLE_S, "4/6 闭合夹爪")
    write_pose(bus, GRASP_LIFT, MOVE_SETTLE_S, "5/6 抬起")
    write_pose(bus, GRASP_HOME, MOVE_SETTLE_S, "6/6 收回待机（夹爪保持闭合）")

    print("抓取动作完成。")
    pos = read_positions(bus)
    print("结束位姿:")
    for name, value in pos.items():
        print(f"  {name}: {value}")


def cmd_list_ports() -> int:
    ports = list_serial_ports()
    if not ports:
        print("未发现串口。请检查 USB 连接，以及是否有 dialout 权限。")
        return 1
    print("可用串口:")
    for p in ports:
        print(f"  - {p}")
    return 0


def _ping_or_read(pkh, ph, sid: int):
    """优先 ping；失败再读 Present_Position(56)。返回 (ok, detail)。"""
    from scservo_sdk import COMM_SUCCESS

    try:
        model, res, err = pkh.ping(ph, sid)
        if res == COMM_SUCCESS:
            return True, f"ping ok model={model} err={err}"
    except Exception:
        pass

    try:
        pos, res, err = pkh.read2ByteTxRx(ph, sid, 56)
        if res == COMM_SUCCESS:
            return True, f"pos={pos} err={err}"
    except Exception:
        pass
    return False, ""


def cmd_scan(
    port: str,
    baud: Optional[int] = None,
    protocol: Optional[int] = None,
) -> int:
    """用 scservo_sdk 扫描 ID；可自动尝试多种波特率/协议。"""
    try:
        from scservo_sdk import PacketHandler, PortHandler
    except ImportError:
        print("缺少 scservo_sdk，请先: pip install feetech-servo-sdk")
        return 1

    describe_port(port)
    bauds = (baud,) if baud is not None else SCAN_BAUDS
    protocols = (protocol,) if protocol is not None else SCAN_PROTOCOLS

    print(f"扫描 {port}")
    print(f"  baud 候选   : {list(bauds)}")
    print(f"  protocol 候选: {list(protocols)}")
    print("  ID 范围     : 0..20（含广播相关常见 ID）")

    any_found = False
    for proto in protocols:
        for rate in bauds:
            print(f"\n--- protocol={proto} baud={rate} ---")
            ph = PortHandler(port)
            pkh = PacketHandler(proto)
            if not ph.openPort():
                print("  打开串口失败（是否被其它进程占用？）")
                continue
            if not ph.setBaudRate(rate):
                print("  设置波特率失败")
                ph.closePort()
                continue

            found = []
            for sid in range(0, 21):
                ok, detail = _ping_or_read(pkh, ph, sid)
                if ok:
                    print(f"  找到 ID={sid}，{detail}")
                    found.append(sid)
            ph.closePort()

            if found:
                any_found = True
                print(f"  => 本组合发现 {len(found)} 个: {found}")
                print(
                    f"\n建议把 start.py 顶部改为: "
                    f"PROTOCOL_VERSION = {proto}  DEFAULT_BAUD = {rate}"
                )
                # 找到一组可用配置即可停止，避免重复刷屏
                return 0

    if not any_found:
        print(
            "\n未扫到任何 Feetech/SCS/STS 舵机。\n"
            "请按下面核对（很常见）:\n"
            "  1) 舵机是否单独供电（5–12V，地线与 USB 适配器 GND 共地）\n"
            "  2) /dev/ttyACM0 是否真是「舵机总线适配器」而不是别的 USB 设备\n"
            "     用: python3 start.py --list-ports  看 description/product\n"
            "  3) 旧 openCF1 臂是 PWM 舵机（#xxxP1500!），不是 Feetech 总线舵机；\n"
            "     若仍是那套臂，本脚本永远扫不到，需要换 STS/SCS 总线臂+USB 转 TTL\n"
            "  4) 再试: python3 start.py --port /dev/ttyUSB0 --scan\n"
        )
        return 1
    return 0


def cmd_grasp(port: str, read_only: bool = False) -> int:
    print(f"连接 {port} ...")
    bus = connect_bus(port)
    torque_on = False
    try:
        pos = read_positions(bus)
        print("当前关节位置(原始 ticks，未标定):")
        for name, value in pos.items():
            print(f"  {name}: {value}")

        if read_only:
            print("只读模式结束。去掉 --read 可执行抓取动作。")
            return 0

        torque_on = True
        run_grasp_sequence(bus)
        return 0
    finally:
        if torque_on:
            print("动作结束，卸力矩...")
            disable_torque_safe(bus)
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass
        print("已断开连接。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="机械臂抓取动作（Feetech / LeRobot）")
    parser.add_argument(
        "--port",
        default=None,
        help="串口路径，例如 /dev/ttyACM0 /dev/ttyUSB0 COM3",
    )
    parser.add_argument("--list-ports", action="store_true", help="列出可用串口")
    parser.add_argument("--scan", action="store_true", help="扫描舵机 ID")
    parser.add_argument(
        "--baud",
        type=int,
        default=None,
        help="指定波特率；不指定则 --scan 会自动尝试常见值",
    )
    parser.add_argument(
        "--protocol",
        type=int,
        choices=(0, 1),
        default=None,
        help="Feetech 协议号 0/1；不指定则 --scan 两种都试",
    )
    parser.add_argument(
        "--read",
        action="store_true",
        help="只读取当前位置，不执行抓取",
    )
    # 兼容旧参数：--demo 等同于执行抓取
    parser.add_argument(
        "--demo",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_ports:
        return cmd_list_ports()

    port = args.port or auto_pick_port()
    if not port:
        print("未找到串口。请用 --list-ports 查看，或手动指定 --port。")
        print("Linux 示例: python3 start.py --port /dev/ttyACM0")
        return 1

    # 防止继续误用 macOS 路径
    if port.startswith("/dev/cu.") and sys.platform.startswith("linux"):
        print(
            f"警告: {port} 是 macOS 设备名，当前是 Linux。"
            "请改用 /dev/ttyACM* 或 /dev/ttyUSB*。"
        )
        return 1

    global PROTOCOL_VERSION, DEFAULT_BAUD
    if args.protocol is not None:
        PROTOCOL_VERSION = args.protocol
    if args.baud is not None:
        DEFAULT_BAUD = args.baud

    if args.scan:
        return cmd_scan(port, baud=args.baud, protocol=args.protocol)

    return cmd_grasp(port, read_only=args.read)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
openCF1 / Km1 机械臂 Ubuntu 串口控制脚本

依赖: pip3 install pyserial
用法:
  python3 opencf1_serial_control.py                  # 交互模式
  python3 opencf1_serial_control.py -p /dev/ttyUSB0  # 指定串口
  python3 opencf1_serial_control.py ping
  python3 opencf1_serial_control.py servo 0 1500 1000
  python3 opencf1_serial_control.py kms 100 50 80 1000
  python3 opencf1_serial_control.py raw '#000P1500T1000!'
"""

from __future__ import annotations

import argparse
import glob
import queue
import sys
import threading
import time
from typing import List, Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("请先安装依赖: pip3 install pyserial")
    sys.exit(1)


DEFAULT_BAUD = 115200
READ_TIMEOUT = 0.05


def list_serial_ports() -> List[str]:
    ports = [p.device for p in list_ports.comports()]
    if ports:
        return ports
    # 兜底：部分系统 list_ports 可能漏检
    candidates = sorted(
        glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/ttyACM*")
        + glob.glob("/dev/ttyS*")
    )
    return candidates


def auto_pick_port() -> Optional[str]:
    ports = list_serial_ports()
    preferred = [p for p in ports if "ttyUSB" in p or "ttyACM" in p]
    if preferred:
        return preferred[0]
    return ports[0] if ports else None


class OpenCF1Arm:
    """openCF1 串口协议封装。指令以 $/#/{/</ 开头，以 ! 或 }/> 结束。"""

    def __init__(self, port: str, baud: int = DEFAULT_BAUD):
        self.port = port
        self.baud = baud
        self.ser: Optional[serial.Serial] = None
        self._reader_stop = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._rx_queue: queue.Queue[str] = queue.Queue()

    def open(self) -> None:
        # 关闭 DTR/RTS，避免 USB-TTL 打开串口时把 STM32 复位
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=READ_TIMEOUT,
            write_timeout=1.0,
            dsrdtr=False,
            rtscts=False,
        )
        try:
            self.ser.dtr = False
            self.ser.rts = False
        except Exception:
            pass
        # 清空上电残留，并给板子一点稳定时间
        time.sleep(0.5)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self._start_reader()
        print(f"[OK] 已打开 {self.port} @ {self.baud}")

    def close(self) -> None:
        self._stop_reader()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[OK] 串口已关闭")

    def _start_reader(self) -> None:
        self._reader_stop.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _stop_reader(self) -> None:
        self._reader_stop.set()
        if self._reader and self._reader.is_alive():
            self._reader.join(timeout=1.0)
        self._reader = None

    def _read_loop(self) -> None:
        assert self.ser is not None
        buf = bytearray()
        while not self._reader_stop.is_set():
            try:
                data = self.ser.read(256)
            except serial.SerialException as exc:
                print(f"\n[ERR] 读串口失败: {exc}")
                break
            if not data:
                # AAA 等短回包可能没有 !/\n，超时后也吐出缓冲
                if buf:
                    text = bytes(buf).decode("utf-8", errors="replace").strip("\r\n")
                    buf.clear()
                    if text:
                        self._rx_queue.put(text)
                        print(f"\n<< {text}", flush=True)
                continue
            buf.extend(data)
            while True:
                cut = -1
                for ch in (b"!", b"\n", b"\r"):
                    i = buf.find(ch)
                    if i >= 0 and (cut < 0 or i < cut):
                        cut = i
                if cut < 0:
                    break
                line = bytes(buf[: cut + 1]).decode("utf-8", errors="replace")
                del buf[: cut + 1]
                text = line.strip("\r\n")
                if text:
                    self._rx_queue.put(text)
                    # 不重打提示符，避免和用户输入抢行导致粘连
                    print(f"\n<< {text}", flush=True)

    def drain_rx(self) -> List[str]:
        items: List[str] = []
        while True:
            try:
                items.append(self._rx_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def wait_rx(self, timeout_s: float = 1.0) -> List[str]:
        deadline = time.time() + timeout_s
        got: List[str] = []
        while time.time() < deadline:
            try:
                got.append(self._rx_queue.get(timeout=0.05))
            except queue.Empty:
                if got:
                    break
        return got

    def send(self, cmd: str, wait_s: float = 0.05) -> None:
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("串口未打开")
        cmd = cmd.strip()
        if not cmd:
            return
        self.drain_rx()
        payload = cmd.encode("ascii", errors="ignore")
        self.ser.write(payload)
        self.ser.flush()
        print(f">> {cmd}")
        if wait_s > 0:
            time.sleep(wait_s)

    # ---------- 高层接口 ----------

    def ping(self) -> bool:
        """握手，板端应回 AAA（可能无结尾符）"""
        self.send("$GETA!", wait_s=0.0)
        replies = self.wait_rx(1.0)
        if not replies:
            print("[WARN] 1秒内无回显。请检查：USART1 接线、TX/RX 交叉、固件、供电")
            return False
        joined = "".join(replies)
        if "AAA" in joined or "$GETA" in joined:
            print("[OK] 握手成功")
            return True
        print(f"[WARN] 有数据但非预期应答: {replies}")
        return False
    def stop_all(self) -> None:
        self.send("$DST!")

    def stop_servo(self, index: int) -> None:
        self.send(f"$DST:{int(index)}!")

    def reset_servos(self) -> None:
        """所有舵机回中"""
        self.send("$DJR!")

    def soft_reset(self) -> None:
        self.send("$RST!")

    def beep(self) -> None:
        self.send("$BEEP!")

    def servo(self, index: int, pwm: int, time_ms: int = 1000) -> None:
        """单舵机: #III PPPP TTTT!  pwm 通常 500~2500"""
        index = int(index)
        pwm = int(pwm)
        time_ms = int(time_ms)
        if not (0 <= index <= 255):
            raise ValueError("舵机号应在 0~255")
        if not (500 <= pwm <= 2500):
            raise ValueError("PWM 建议在 500~2500")
        if not (0 <= time_ms <= 9999):
            raise ValueError("时间应在 0~9999 ms")
        self.send(f"#{index:03d}P{pwm:04d}T{time_ms:04d}!")

    def multi_servo(self, items: List[tuple], wrap: bool = True) -> None:
        """
        多舵机: items = [(index, pwm, time_ms), ...]
        wrap=True 时用 {#..!#..!} 包裹（板端 uart_mode=3）
        """
        parts = []
        for index, pwm, time_ms in items:
            parts.append(f"#{int(index):03d}P{int(pwm):04d}T{int(time_ms):04d}!")
        body = "".join(parts)
        self.send(f"{{{body}}}" if wrap else body)

    def kinematics(self, x: float, y: float, z: float, time_ms: int = 1000) -> None:
        """笛卡尔坐标(mm)，y 必须 >= 0"""
        if y < 0:
            raise ValueError("固件要求 y >= 0")
        self.send(f"$KMS:{x:.1f},{y:.1f},{z:.1f},{int(time_ms)}!")

    def do_group(self, index: int) -> None:
        self.send(f"$DGS:{int(index)}!")

    def do_group_range(self, start: int, end: int, times: int = 1) -> None:
        self.send(f"$DGT:{int(start)}-{int(end)},{int(times)}!")

    def smart_mode(self, mode: int) -> None:
        self.send(f"$SMODE{int(mode)}!")


HELP_TEXT = """
可用命令:
  help                         显示帮助
  ports                        列出串口
  ping                         握手 $GETA!
  stop                         全部停止 $DST!
  stop <id>                    停止指定舵机
  home                         舵机复位 $DJR!
  beep                         蜂鸣
  servo <id> <pwm> [time_ms]   单舵机，如: servo 0 1500 1000
  multi <id:pwm:t> ...         多舵机，如: multi 0:1500:1000 1:1200:1000
  kms <x> <y> <z> [time_ms]    逆解坐标(mm)，如: kms 100 50 80 1000
  group <n>                    执行动作组 n
  groups <a> <b> [times]       执行动作组 a~b
  smode <n>                    智能模式 $SMODEn!
  raw <cmd>                    发送原始指令（可含空格外的整段）
  quit / exit                  退出
"""


def run_interactive(arm: OpenCF1Arm) -> None:
    print(HELP_TEXT)
    print("输入 help 查看命令。Ctrl+C 或 quit 退出。")
    while True:
        try:
            line = input("opencf1> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if not handle_line(arm, line):
            break


def handle_line(arm: OpenCF1Arm, line: str) -> bool:
    """返回 False 表示退出。"""
    parts = line.split()
    cmd = parts[0].lower()

    try:
        if cmd in ("quit", "exit", "q"):
            return False
        if cmd == "help":
            print(HELP_TEXT)
        elif cmd == "ports":
            ports = list_serial_ports()
            print("可用串口:" if ports else "未发现串口")
            for p in ports:
                print(f"  {p}")
        elif cmd == "ping":
            arm.ping()
        elif cmd == "stop":
            if len(parts) == 1:
                arm.stop_all()
            else:
                arm.stop_servo(int(parts[1]))
        elif cmd == "home":
            arm.reset_servos()
        elif cmd == "beep":
            arm.beep()
        elif cmd == "servo":
            if len(parts) != 3 and len(parts) != 4:
                print("用法: servo <id> <pwm> [time_ms]   例: servo 0 1500 1000")
            else:
                time_ms = int(parts[3]) if len(parts) == 4 else 1000
                arm.servo(int(parts[1]), int(parts[2]), time_ms)
        elif cmd == "multi":
            if len(parts) < 2:
                print("用法: multi <id:pwm:t> [id:pwm:t ...]")
            else:
                items = []
                for token in parts[1:]:
                    a, b, c = token.split(":")
                    items.append((int(a), int(b), int(c)))
                arm.multi_servo(items)
        elif cmd == "kms":
            if len(parts) < 4:
                print("用法: kms <x> <y> <z> [time_ms]")
            else:
                time_ms = int(parts[4]) if len(parts) > 4 else 1000
                arm.kinematics(float(parts[1]), float(parts[2]), float(parts[3]), time_ms)
        elif cmd == "group":
            if len(parts) < 2:
                print("用法: group <n>")
            else:
                arm.do_group(int(parts[1]))
        elif cmd == "groups":
            if len(parts) < 3:
                print("用法: groups <start> <end> [times]")
            else:
                times = int(parts[3]) if len(parts) > 3 else 1
                arm.do_group_range(int(parts[1]), int(parts[2]), times)
        elif cmd == "smode":
            if len(parts) < 2:
                print("用法: smode <n>")
            else:
                arm.smart_mode(int(parts[1]))
        elif cmd == "raw":
            raw = line[len(parts[0]) :].strip()
            if not raw:
                print("用法: raw <command>")
            else:
                arm.send(raw)
        else:
            # 允许直接粘贴固件原生指令
            if line.startswith(("$", "#", "{", "<")):
                arm.send(line)
            else:
                print(f"未知命令: {cmd}，输入 help 查看帮助")
    except (ValueError, RuntimeError) as exc:
        print(f"[ERR] {exc}")
    except serial.SerialException as exc:
        print(f"[ERR] 串口异常: {exc}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="openCF1 机械臂 Ubuntu 串口控制",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="不带子命令则进入交互模式。",
    )
    parser.add_argument("-p", "--port", help="串口设备，如 /dev/ttyUSB0")
    parser.add_argument("-b", "--baud", type=int, default=DEFAULT_BAUD, help="波特率，默认 115200")
    parser.add_argument("--list", action="store_true", help="仅列出串口后退出")

    sub = parser.add_subparsers(dest="action")

    sub.add_parser("ping", help="握手")
    sub.add_parser("stop", help="全部停止")
    sub.add_parser("home", help="舵机复位")
    sub.add_parser("beep", help="蜂鸣")

    p_servo = sub.add_parser("servo", help="单舵机控制")
    p_servo.add_argument("id", type=int)
    p_servo.add_argument("pwm", type=int)
    p_servo.add_argument("time_ms", type=int, nargs="?", default=1000)

    p_kms = sub.add_parser("kms", help="笛卡尔逆解")
    p_kms.add_argument("x", type=float)
    p_kms.add_argument("y", type=float)
    p_kms.add_argument("z", type=float)
    p_kms.add_argument("time_ms", type=int, nargs="?", default=1000)

    p_group = sub.add_parser("group", help="执行动作组")
    p_group.add_argument("n", type=int)

    p_groups = sub.add_parser("groups", help="执行动作组区间")
    p_groups.add_argument("start", type=int)
    p_groups.add_argument("end", type=int)
    p_groups.add_argument("times", type=int, nargs="?", default=1)

    p_raw = sub.add_parser("raw", help="发送原始指令")
    p_raw.add_argument("cmd", nargs=argparse.REMAINDER)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        ports = list_serial_ports()
        if not ports:
            print("未发现串口设备")
            return 1
        for p in ports:
            print(p)
        return 0

    port = args.port or auto_pick_port()
    if not port:
        print("未找到串口。请用 -p /dev/ttyUSB0 指定，或先执行 --list")
        print("提示: sudo usermod -aG dialout $USER 后重新登录")
        return 1

    arm = OpenCF1Arm(port, args.baud)
    try:
        arm.open()
    except serial.SerialException as exc:
        print(f"[ERR] 无法打开 {port}: {exc}")
        print("检查: 设备是否插好、权限是否在 dialout 组、是否被其他程序占用")
        return 1

    try:
        action = args.action
        if action is None:
            run_interactive(arm)
        elif action == "ping":
            ok = arm.ping()
            return 0 if ok else 2
        elif action == "stop":
            arm.stop_all()
        elif action == "home":
            arm.reset_servos()
        elif action == "beep":
            arm.beep()
        elif action == "servo":
            arm.servo(args.id, args.pwm, args.time_ms)
            time.sleep(max(args.time_ms / 1000.0, 0.1))
        elif action == "kms":
            arm.kinematics(args.x, args.y, args.z, args.time_ms)
            time.sleep(max(args.time_ms / 1000.0, 0.1))
        elif action == "group":
            arm.do_group(args.n)
            time.sleep(0.5)
        elif action == "groups":
            arm.do_group_range(args.start, args.end, args.times)
            time.sleep(0.5)
        elif action == "raw":
            cmd = " ".join(args.cmd).strip()
            if not cmd:
                print("用法: raw <command>")
                return 1
            arm.send(cmd)
            time.sleep(0.3)
        return 0
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except (ValueError, RuntimeError) as exc:
        print(f"[ERR] {exc}")
        return 1
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())

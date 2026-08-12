#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STM32(openCF1) 机械臂串口连通诊断 — Windows 版。

正确标志:
  发 $GETA! 后必须收到 AAA，且舵机应有动作。
  只有 TX、无 RX、臂不动 = 硬件链路未通，不是代码逻辑问题。

【经验结论 / 勿改 open_port】
  打开时序必须保持:
    open -> sleep(0.5) -> dtr=False, rts=False -> sleep(0.4) -> 清缓冲
  对本板：翻转/拉高 DTR（所谓软复位）反而会把单片机弄进异常状态，
  表现与「先插计算盒子再插电脑就不通」相同。不要再加 DTR 脉冲。

用法:
  python test_win.py --list
  python test_win.py --diagnose
  python test_win.py --cmd '$GETA!'
  python test_win.py --port COM7 --baud 115200 --cmd '$GETA!'
"""

from __future__ import print_function

import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("缺少 pyserial，请先安装:")
    print("  pip install pyserial")
    sys.exit(1)


def list_serial_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("未发现串口设备")
        return
    print("可用串口:")
    for p in ports:
        print("  {}  {}  [{}]".format(p.device, p.description, p.hwid))


def open_port(port, baud):
    # 勿改此时序；与 Linux test1.py 对齐（Linux 仅额外做 termios）
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.timeout = 1.2
    ser.write_timeout = 1.0
    ser.rtscts = False
    ser.xonxoff = False
    ser.dsrdtr = False

    ser.open()
    time.sleep(0.5)
    ser.dtr = False
    ser.rts = False
    time.sleep(0.4)

    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.2)
    return ser


def read_for(ser, wait_s):
    deadline = time.time() + wait_s
    chunks = []
    while time.time() < deadline:
        n = ser.in_waiting
        if n:
            chunks.append(ser.read(n))
        else:
            time.sleep(0.02)
    return b"".join(chunks)


def send(ser, cmd, wait_s=1.5):
    data = cmd.encode("ascii")
    written = ser.write(data)
    ser.flush()
    time.sleep(0.15)
    print("TX: {}  ({} bytes, hex={})".format(cmd, written, data.hex()))
    resp = read_for(ser, wait_s)
    if resp:
        text = resp.decode("ascii", errors="replace")
        print("RX: {!r}  (hex={})".format(text, resp.hex()))
    else:
        print("RX: (无数据)")
    return resp


def print_checklist():
    print("")
    print("==== 硬件排查清单（当前判定：链路未通）====")
    print("【封装 USB】电脑 USB -> 板子 USB-TTL；电源独立，换线不会自动复位 MCU")
    print("1. 电源 ON，PWR/MCU 灯亮；必要时按 RST 后再测")
    print("2. 设备管理器确认 COM 口（CH340），可用 --list 查看")
    print("3. 波特率 115200")
    print("4. 打开时序：dtr=False, rts=False（禁止 DTR 脉冲）")
    print("5. 若刚用异常版脚本测过，先按 RST 再测")
    print("6. Linux 对照: python3 test_lin.py --port /dev/ttyUSB0 --cmd '$GETA!'")


def diagnose(ser):
    print("--- 诊断开始 ---")
    ok = False

    print("监听板上主动输出 1s ...")
    boot = read_for(ser, 1.0)
    if boot:
        print("RX(idle): {!r} hex={}".format(
            boot.decode("ascii", errors="replace"), boot.hex()))
    else:
        print("RX(idle): (无数据) —— 收线/口位可能不对，或板子未运行")

    for cmd, wait_s in (
        ("$GETA!", 1.0),
        ("#000P1000T1000!", 1.5),
        ("#000P2000T1000!", 1.5),
        ("#000P1500T1000!", 1.5),
    ):
        resp = send(ser, cmd, wait_s=wait_s)
        if resp and b"AAA" in resp:
            ok = True

    if ok:
        print("结果: 串口应答正常。若仍不动，查舵机供电/舵机号/PWM 口。")
    else:
        print("结果: 未收到 AAA，且大概率指令未进 STM32。")
        print_checklist()
    print("--- 诊断结束 ---")
    return 0 if ok else 2


def main():
    parser = argparse.ArgumentParser(description="STM32 arm serial diagnose (Windows)")
    parser.add_argument("--port", default="COM7", help="串口设备，如 COM7")
    parser.add_argument("--baud", type=int, default=115200, help="波特率")
    parser.add_argument("--list", action="store_true", help="列出串口后退出")
    parser.add_argument("--diagnose", action="store_true", help="完整诊断")
    parser.add_argument("--cmd", default="", help="自定义单条指令")
    args = parser.parse_args()

    if args.list:
        list_serial_ports()
        return 0

    try:
        ser = open_port(args.port, args.baud)
    except serial.SerialException as e:
        print("打开串口失败: {}".format(e))
        print("可先执行: python test_win.py --list")
        return 1

    print("已打开 {} @ {}".format(args.port, args.baud))

    try:
        if args.diagnose or not args.cmd:
            return diagnose(ser)
        resp = send(ser, args.cmd, wait_s=1.0)
        if not resp:
            print_checklist()
            return 2
        return 0
    finally:
        ser.close()
        print("串口已关闭")


if __name__ == "__main__":
    sys.exit(main())

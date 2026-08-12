#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STM32(openCF1) arm serial diagnose - Linux.

OK flag: send $GETA! and receive AAA.

Linux open_port (CH340):
  stty -hupcl -> open -> dtr/rts=False immediately -> ioctl force low
  -> clear HUPCL -> sleep -> flush. Do NOT leave DTR high after open.

Usage:
  python3 test_lin.py --list
  python3 test_lin.py --diagnose
  python3 test_lin.py --cmd '$GETA!'
  python3 test_lin.py --port /dev/ttyUSB0 --baud 115200 --cmd '$GETA!'
  python3 test_lin.py --cmd '$GETA!' --output test_lin.log   # also write to file
"""

from __future__ import print_function

import argparse
import contextlib
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("Missing pyserial. Install: pip3 install pyserial")
    print("Or: sudo apt install python3-serial")
    sys.exit(1)

try:
    import termios
except ImportError:
    print("This script needs Linux (termios). On Windows use test_win.py")
    sys.exit(1)


def list_serial_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found")
        return
    print("Serial ports:")
    for p in ports:
        print("  {}  {}  [{}]".format(p.device, p.description, p.hwid))


def _clear_hupcl(fd):
    """Clear HUPCL so close/open does not drop DTR and soft-reset STM32."""
    attrs = termios.tcgetattr(fd)
    attrs[2] &= ~termios.HUPCL
    attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ECHOE | termios.ECHONL | termios.ISIG)
    attrs[0] &= ~(termios.ICRNL | termios.INLCR | termios.IGNCR | termios.ISTRIP)
    attrs[1] &= ~termios.ONLCR
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def _force_lines_low(fd):
    """Force DTR/RTS low via ioctl (stronger than pyserial attrs alone)."""
    import array
    import fcntl

    # Linux tty ioctl numbers
    tiocmget = getattr(termios, "TIOCMGET", 0x5415)
    tiocmset = getattr(termios, "TIOCMSET", 0x5418)
    tiocm_dtr = getattr(termios, "TIOCM_DTR", 0x002)
    tiocm_rts = getattr(termios, "TIOCM_RTS", 0x004)

    status = array.array("i", [0])
    fcntl.ioctl(fd, tiocmget, status, True)
    status[0] &= ~(tiocm_dtr | tiocm_rts)
    fcntl.ioctl(fd, tiocmset, status, True)
    return status[0]


def open_port(port, baud):
    # Linux CH340 often asserts DTR on open; that soft-resets this STM32 into a
    # dead state (Windows OK, Linux TX-only / no AAA). Mitigate after open.
    import subprocess

    # Prefer stty before pyserial open (avoids an extra open/close pulse).
    try:
        subprocess.call(
            ["stty", "-F", port, "-hupcl"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass

    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.timeout = 0.2
    ser.write_timeout = 1.0
    ser.rtscts = False
    ser.xonxoff = False
    ser.dsrdtr = False

    ser.open()
    # Lower lines immediately; do not wait with DTR still high.
    try:
        ser.dtr = False
        ser.rts = False
    except Exception:
        pass
    try:
        line_status = _force_lines_low(ser.fd)
        print("line_status_after_low: 0x{:x} (DTR/RTS forced low)".format(line_status))
    except Exception as e:
        print("force_lines_low failed: {}".format(e))
    _clear_hupcl(ser.fd)
    time.sleep(0.5)

    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.2)
    return ser


def read_for(ser, wait_s):
    # Prefer blocking read(timeout) over only polling in_waiting (unreliable on
    # some Linux USB-serial stacks).
    deadline = time.time() + wait_s
    chunks = []
    while time.time() < deadline:
        n = ser.in_waiting
        chunk = ser.read(n if n else 1)
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


def send(ser, cmd, wait_s=2.0):
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
        print("RX: (no data)")
    return resp


def print_checklist():
    print("")
    print("==== Checklist (no AAA) ====")
    print("1. Power ON, PWR/MCU LED on; press RST if needed")
    print("2. Use /dev/ttyUSB* (CH340), NOT /dev/ttyACM0 (DJI)")
    print("3. Baud 115200")
    print("4. open_port: dtr=False, rts=False (no DTR pulse)")
    print("5. Exit script after test; press RST if previous bad script ran")
    print("6. Windows check: python test_win.py --port COM7 --cmd '$GETA!'")


def diagnose(ser):
    print("--- diagnose start ---")
    ok = False

    print("listen idle 1s ...")
    boot = read_for(ser, 1.0)
    if boot:
        print("RX(idle): {!r} hex={}".format(
            boot.decode("ascii", errors="replace"), boot.hex()))
    else:
        print("RX(idle): (no data)")

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
        print("Result: serial OK (got AAA).")
    else:
        print("Result: no AAA.")
        print_checklist()
    print("--- diagnose end ---")
    return 0 if ok else 2


class _Tee(object):
    """Write to terminal and optionally a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser(description="STM32 arm serial diagnose (Linux)")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="serial device")
    parser.add_argument("--baud", type=int, default=115200, help="baud rate")
    parser.add_argument("--list", action="store_true", help="list ports and exit")
    parser.add_argument("--diagnose", action="store_true", help="full diagnose")
    parser.add_argument("--cmd", default="", help="single command")
    parser.add_argument(
        "--output", default="",
        help="also append output to this file (default: terminal only)",
    )
    parser.add_argument(
        "--rst-wait", type=float, default=0.0,
        help="seconds after open to press STM32 RST before TX (e.g. 5)",
    )
    args = parser.parse_args()

    if not args.output:
        print("\n=== {} ===".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        sys.stdout.flush()
        return run(args)

    with open(args.output, "a", encoding="utf-8", buffering=1) as logf:
        tee = _Tee(sys.__stdout__, logf)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print("\n=== {} ===".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            return run(args)


def run(args):
    if args.list:
        list_serial_ports()
        return 0

    try:
        ser = open_port(args.port, args.baud)
    except serial.SerialException as e:
        print("Open failed: {}".format(e))
        print("Try: python3 test_lin.py --list")
        return 1

    print("Opened {} @ {}".format(args.port, args.baud))

    if getattr(args, "rst_wait", 0) > 0:
        print("Press STM32 RST NOW, waiting {:.1f}s before TX ...".format(
            args.rst_wait))
        time.sleep(args.rst_wait)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

    try:
        if args.diagnose or not args.cmd:
            return diagnose(ser)
        resp = send(ser, args.cmd, wait_s=2.0)
        if not resp:
            print_checklist()
            return 2
        return 0
    finally:
        ser.close()
        print("Closed")


if __name__ == "__main__":
    sys.exit(main())

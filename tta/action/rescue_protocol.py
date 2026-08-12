# 小车 ↔ 无人机 通信协议（ArmPi Pro 适配版）
# --skip-drone 时可无 fabric2；联机模式才需要: pip install fabric2

from __future__ import print_function

import csv
import io
import time


class RescueOrder(object):
    def __init__(self, zone, level, all_orders=None):
        self.zone = int(zone)
        self.level = int(level)
        self.all_orders = list(all_orders) if all_orders else []


def _ssh_connect(host, user, password):
    try:
        from fabric2 import Connection
    except ImportError as exc:
        raise ImportError(
            "联机通讯需要 fabric2，请先安装: pip3 install fabric2"
        ) from exc
    return Connection(host=host, user=user, connect_kwargs={"password": password})


def _parse_zone(row):
    raw = row.get("target_zone")
    if raw is None or str(raw).strip() == "":
        raw = row.get("救援点", "0")
    text = str(raw).strip().replace("救援点", "")
    return int(text)


def _parse_level(row):
    raw = row.get("level")
    if raw is None or str(raw).strip() == "":
        raw = row.get("救援等级", "1")
    text = str(raw).strip().replace("级", "")
    return int(text)


def parse_rescue_cmd_csv(csv_text, preferred_level=1):
    text = (csv_text or "").strip()
    if not text:
        return None, []
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return None, []

    orders = []
    for row in reader:
        if not row or all(v is None or str(v).strip() == "" for v in row.values()):
            continue
        try:
            orders.append(
                RescueOrder(zone=_parse_zone(row), level=_parse_level(row))
            )
        except (TypeError, ValueError):
            continue

    if not orders:
        return None, []

    selected = None
    for order in orders:
        if order.level == int(preferred_level):
            selected = RescueOrder(
                zone=order.zone,
                level=order.level,
                all_orders=list(orders),
            )
            break
    return selected, orders


def parse_rescue_target_flag(flag_text):
    value = (flag_text or "").strip()
    if value.startswith("\ufeff"):
        value = value.lstrip("\ufeff")
    if not value:
        raise ValueError("empty rescue_target.flag")
    value = value.splitlines()[0].strip()
    try:
        zone = int(value)
    except ValueError as exc:
        raise ValueError("Invalid rescue_target.flag content: %r" % value) from exc
    if zone not in (1, 2, 3, 4):
        raise ValueError("Invalid target zone: %s" % zone)
    return zone


def wait_for_rescue_target(
    remote_path="/mnt/rescue_target.flag",
    host="192.168.31.110",
    user="root",
    password="123456",
    timeout=120,
    poll_interval=1.0,
):
    """阻塞等待无人机写入目标点 flag（内容为 1/2/3/4）。"""
    deadline = None if timeout is None else time.time() + timeout
    while True:
        try:
            with _ssh_connect(host, user, password) as conn:
                result = conn.run("test -e %s" % remote_path, hide=True, warn=True)
                if not result.ok:
                    if deadline is not None and time.time() >= deadline:
                        return None
                    time.sleep(poll_interval)
                    continue
                flag_text = conn.run("cat %s" % remote_path, hide=True).stdout
                try:
                    zone = parse_rescue_target_flag(flag_text)
                except ValueError as exc:
                    print(
                        "wait_for_rescue_target: %s; content=%r"
                        % (exc, (flag_text or "").strip())
                    )
                    return None
                print("wait_for_rescue_target: zone=%s" % zone)
                conn.run("rm -f %s" % remote_path, hide=True, warn=True)
                return RescueOrder(zone=zone, level=0)
        except Exception as exc:
            print("wait_for_rescue_target retry: %s" % exc)
            time.sleep(2)
        if deadline is not None and time.time() >= deadline:
            return None


def notify_loading_done(
    remote_path="/mnt/loading_done.flag",
    host="192.168.31.110",
    user="root",
    password="123456",
):
    """通知无人机：装货完成。"""
    while True:
        try:
            with _ssh_connect(host, user, password) as conn:
                result = conn.run("touch %s" % remote_path, hide=True)
                if result.ok:
                    return True
        except Exception as exc:
            print("notify_loading_done retry: %s" % exc)
        time.sleep(2)


def wait_for_delivery_done(
    remote_path="/mnt/delivery_done.flag",
    host="192.168.31.110",
    user="root",
    password="123456",
    timeout=300,
):
    """等待无人机投送完成。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with _ssh_connect(host, user, password) as conn:
                result = conn.run("test -e %s" % remote_path, hide=True, warn=True)
                if result.ok:
                    conn.run("rm -f %s" % remote_path, hide=True, warn=True)
                    return True
        except Exception as exc:
            print("wait_for_delivery_done retry: %s" % exc)
        time.sleep(2)
    return False


def notify_unload_done(
    remote_path="/mnt/unload_done.flag",
    host="192.168.31.110",
    user="root",
    password="123456",
):
    """通知无人机：小车卸货完成。"""
    while True:
        try:
            with _ssh_connect(host, user, password) as conn:
                result = conn.run("touch %s" % remote_path, hide=True)
                if result.ok:
                    return True
        except Exception as exc:
            print("notify_unload_done retry: %s" % exc)
        time.sleep(2)

# 小车 ↔ 无人机 通信协议

import csv
import os
import time
from dataclasses import dataclass

from fabric2 import Connection


@dataclass
class RescueOrder:
    zone: int
    level: int


def _ssh_connect(host, user, password):
    return Connection(host=host, user=user, connect_kwargs={"password": password})


def wait_for_rescue_cmd(
    remote_path="/mnt/rescue_cmd.csv",
    host="192.168.31.110",
    user="root",
    password="123456",
    timeout=120,
):
    """阻塞等待无人机写入救援指令 CSV，返回 RescueOrder 或 None。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with _ssh_connect(host, user, password) as conn:
                result = conn.run(f"test -e {remote_path}", hide=True, warn=True)
                if not result.ok:
                    time.sleep(1)
                    continue
                csv_text = conn.run(f"cat {remote_path}", hide=True).stdout
            rows = list(csv.DictReader(csv_text.strip().splitlines()))
            if not rows:
                time.sleep(1)
                continue
            row = rows[0]
            zone = int(row.get("target_zone") or row.get("救援点", "0").replace("救援点", ""))
            level_raw = row.get("level") or row.get("救援等级", "1")
            level = int(str(level_raw).replace("级", ""))
            return RescueOrder(zone=zone, level=level)
        except Exception as exc:
            print(f"wait_for_rescue_cmd retry: {exc}")
            time.sleep(2)
    return None


def notify_loading_done(
    remote_path="/mnt/loading_done.flag",
    host="192.168.31.110",
    user="root",
    password="123456",
):
    """通知无人机：装货完成，可以起飞。"""
    while True:
        try:
            with _ssh_connect(host, user, password) as conn:
                result = conn.run(f"touch {remote_path}", hide=True)
                if result.ok:
                    return True
        except Exception as exc:
            print(f"notify_loading_done retry: {exc}")
        time.sleep(2)


def wait_for_delivery_done(
    remote_path="/mnt/delivery_done.flag",
    host="192.168.31.110",
    user="root",
    password="123456",
    timeout=300,
):
    """等待无人机在救援区卸货完成。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with _ssh_connect(host, user, password) as conn:
                result = conn.run(f"test -e {remote_path}", hide=True, warn=True)
                if result.ok:
                    conn.run(f"rm -f {remote_path}", hide=True, warn=True)
                    return True
        except Exception as exc:
            print(f"wait_for_delivery_done retry: {exc}")
        time.sleep(2)
    return False

# 小车 ↔ 无人机 通信协议

import csv
import io
import time
from dataclasses import dataclass, field
from typing import List

from fabric2 import Connection


@dataclass
class RescueOrder:
    zone: int
    level: int
    # CSV 中解析出的全部 (zone, level)，便于日志/多轮扩展
    all_orders: List["RescueOrder"] = field(default_factory=list, repr=False)


def _ssh_connect(host, user, password):
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
    """
    解析 rescue_cmd.csv 全部行，按文件顺序选择第一个
    level == preferred_level 的救援点（多条 level=1 时只取第一条）。

    期望格式示例：
        target_zone,level
        1,2
        2,1
        3,1
        4,2
    则 preferred_level=1 时返回 zone=2, level=1（忽略后面的 zone=3）。
    """
    text = (csv_text or "").strip()
    if not text:
        return None, []

    # 兼容 Excel UTF-8 BOM
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

    # 按 CSV 行序，只取第一个匹配 preferred_level 的救援点
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
    """
    解析 rescue_target.flag 文本，返回目标区号 1/2/3/4。
    文件内容仅为单个数字，非法内容抛出 ValueError。
    """
    value = (flag_text or "").strip()
    if value.startswith("\ufeff"):
        value = value.lstrip("\ufeff")
    if not value:
        raise ValueError("empty rescue_target.flag")
    value = value.splitlines()[0].strip()
    try:
        zone = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid rescue_target.flag content: {value!r}") from exc
    if zone not in (1, 2, 3, 4):
        raise ValueError(f"Invalid target zone: {zone}")
    return zone


def wait_for_rescue_target(
    remote_path="/mnt/rescue_target.flag",
    host="192.168.31.110",
    user="root",
    password="123456",
    timeout=120,
    poll_interval=1.0,
):
    """
    阻塞等待无人机写入最终目标点 flag（内容为单个数字 1/2/3/4）。
    读取成功后删除远程文件，返回 RescueOrder(zone=N, level=0)。
    内容非法时记录错误并返回 None，不回退解析 rescue_cmd.csv。
    """
    deadline = None if timeout is None else time.time() + timeout
    while True:
        try:
            with _ssh_connect(host, user, password) as conn:
                result = conn.run(f"test -e {remote_path}", hide=True, warn=True)
                if not result.ok:
                    if deadline is not None and time.time() >= deadline:
                        return None
                    time.sleep(poll_interval)
                    continue
                flag_text = conn.run(f"cat {remote_path}", hide=True).stdout
                try:
                    zone = parse_rescue_target_flag(flag_text)
                except ValueError as exc:
                    print(
                        "wait_for_rescue_target: %s; content=%r"
                        % (exc, (flag_text or "").strip())
                    )
                    return None
                print("wait_for_rescue_target: zone=%s" % zone)
                conn.run(f"rm -f {remote_path}", hide=True, warn=True)
                return RescueOrder(zone=zone, level=0)
        except Exception as exc:
            print(f"wait_for_rescue_target retry: {exc}")
            time.sleep(2)
        if deadline is not None and time.time() >= deadline:
            return None


def wait_for_rescue_cmd(
    remote_path="/mnt/rescue_cmd.csv",
    host="192.168.31.110",
    user="root",
    password="123456",
    timeout=120,
    preferred_level=1,
):
    """
    [已废弃主流程] 阻塞等待无人机写入救援指令 CSV。
    主任务请改用 wait_for_rescue_target() 读取 /mnt/rescue_target.flag。
    本函数仅保留给调试/兼容旧脚本。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with _ssh_connect(host, user, password) as conn:
                result = conn.run(f"test -e {remote_path}", hide=True, warn=True)
                if not result.ok:
                    time.sleep(1)
                    continue
                csv_text = conn.run(f"cat {remote_path}", hide=True).stdout
                selected, orders = parse_rescue_cmd_csv(
                    csv_text, preferred_level=preferred_level
                )
                if not orders:
                    time.sleep(1)
                    continue
                if selected is None:
                    print(
                        "wait_for_rescue_cmd: CSV 有 %d 行，但没有 level=%s；内容=%r"
                        % (len(orders), preferred_level, csv_text.strip())
                    )
                    time.sleep(1)
                    continue
                print(
                    "wait_for_rescue_cmd: 读到 %d 条 -> 取第一个 level=%s 的 zone=%s；全部=%s"
                    % (
                        len(orders),
                        preferred_level,
                        selected.zone,
                        [(o.zone, o.level) for o in orders],
                    )
                )
                conn.run(f"rm -f {remote_path}", hide=True, warn=True)
                return selected
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


def notify_unload_done(
    remote_path="/mnt/unload_done.flag",
    host="192.168.31.110",
    user="root",
    password="123456",
):
    """通知无人机：小车已在救援区卸货完成。"""
    while True:
        try:
            with _ssh_connect(host, user, password) as conn:
                result = conn.run(f"touch {remote_path}", hide=True)
                if result.ok:
                    return True
        except Exception as exc:
            print(f"notify_unload_done retry: {exc}")
        time.sleep(2)

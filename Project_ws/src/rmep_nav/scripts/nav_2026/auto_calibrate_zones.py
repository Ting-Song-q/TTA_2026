#!/usr/bin/python3
# coding=UTF-8
"""
半自动现场标定向导：减少逐区手工敲命令与重复 RViz 初定位。

典型用法（ROS 栈已启动：roscore + rmep_base + map_amcl_move）:

  # 首次：在停车区执行向导，标定全部区域
  python3 auto_calibrate_zones.py --write

  # 之后每次上电：用已标定的 parking 自动发布 /initialpose
  python3 auto_calibrate_zones.py --init-only

  # 仅有粗略坐标时，先自动导航再人工微调后记录
  python3 auto_calibrate_zones.py --navigate --write

流程:
  1. （可选）自动发布 parking 初定位
  2. 按顺序引导 parking → pickup → loading → rescue 1-4
  3. 每步：可选 move_base 导航 → 等待确认 → 自动采样写入 zone_calibration.yaml
"""

import argparse
import sys
from pathlib import Path

import actionlib
import rospy
import tf
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler

from calibration_utils import (
    apply_auto_initial_pose,
    calibration_path,
    is_navigable_pose,
    record_zone_pose,
    update_zone_overlay,
    zone_pose_from_config,
)
from config_loader import load_mission_config, load_yaml, save_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "mission_config.yaml"

DEFAULT_ZONE_STEPS = [
    ("parking", None, "停车区：车辆停放在返航安全位"),
    ("pickup", None, "取货区：四轮尽量进入黄线框"),
    ("loading", None, "装货区：四轮尽量进入黄线框"),
    ("rescue", 1, "救援区 1"),
    ("rescue", 2, "救援区 2"),
    ("rescue", 3, "救援区 3"),
    ("rescue", 4, "救援区 4"),
]


def _zone_label(zone_name, zone_id):
    if zone_name == "rescue":
        return "rescue_%d" % zone_id
    return zone_name


def _make_move_base_goal(x, y, yaw):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = float(x)
    goal.target_pose.pose.position.y = float(y)
    q = quaternion_from_euler(0, 0, float(yaw))
    goal.target_pose.pose.orientation.x = q[0]
    goal.target_pose.pose.orientation.y = q[1]
    goal.target_pose.pose.orientation.z = q[2]
    goal.target_pose.pose.orientation.w = q[3]
    return goal


def _navigate_to_pose(config, pose, timeout=120.0):
    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        raise RuntimeError("move_base action server unavailable")

    goal = _make_move_base_goal(pose[0], pose[1], pose[2])
    client.send_goal(goal)
    finished = client.wait_for_result(rospy.Duration(timeout))
    state = client.get_state()
    if not finished or state != GoalStatus.SUCCEEDED:
        raise RuntimeError("navigation failed state=%s" % state)


def _parse_zone_steps(selection):
    if not selection:
        return list(DEFAULT_ZONE_STEPS)
    mapping = {_zone_label(name, zone_id): (name, zone_id, hint) for name, zone_id, hint in DEFAULT_ZONE_STEPS}
    steps = []
    for item in selection:
        key = item.strip()
        if key not in mapping:
            raise ValueError("unknown zone: %s" % key)
        name, zone_id, hint = mapping[key]
        steps.append((name, zone_id, hint))
    return steps


def _prompt_continue(label, navigate=False):
    if navigate:
        prompt = "[%s] 已尝试自动导航。请人工微调到安全位，按 Enter 记录（s 跳过，q 退出）: "
    else:
        prompt = "[%s] 请将车辆移动到安全位，按 Enter 记录（s 跳过，q 退出）: "
    try:
        answer = input(prompt % label).strip().lower()
    except EOFError:
        return "quit"
    if answer in ("q", "quit", "exit"):
        return "quit"
    if answer in ("s", "skip"):
        return "skip"
    return "record"


def run_wizard(
    config,
    overlay,
    steps,
    listener,
    navigate=False,
    samples=10,
    interval=0.10,
    max_spread=0.03,
    force=False,
    nav_timeout=120.0,
):
    records = []
    for zone_name, zone_id, hint in steps:
        label = _zone_label(zone_name, zone_id)
        print("\n=== %s ===" % label)
        print(hint)

        pose = zone_pose_from_config(config, zone_name, zone_id)
        if navigate and is_navigable_pose(pose):
            try:
                print("导航至 (%.3f, %.3f, %.3f) ..." % pose)
                _navigate_to_pose(config, pose, timeout=nav_timeout)
                rospy.sleep(1.0)
            except RuntimeError as exc:
                print("导航失败: %s（请手动到位）" % exc)

        action = _prompt_continue(label, navigate=navigate)
        if action == "quit":
            break
        if action == "skip":
            print("跳过 %s" % label)
            continue

        result = record_zone_pose(
            config,
            zone_name,
            zone_id=zone_id,
            samples=samples,
            interval=interval,
            max_spread=max_spread,
            force=force,
            sync_bounds=True,
            listener=listener,
        )
        pose_data = update_zone_overlay(
            overlay,
            zone_name,
            zone_id,
            result["pose"],
            sync_bounds=True,
        )
        records.append((label, pose_data, result))
        print(
            "已记录 %s pose=%s spread=%.3f"
            % (label, pose_data, result["spread"])
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Guided semi-automatic zone calibration")
    parser.add_argument(
        "--zones",
        nargs="+",
        metavar="ZONE",
        help="subset: parking pickup loading rescue_1 ... rescue_4",
    )
    parser.add_argument("--write", action="store_true", help="save zone_calibration.yaml")
    parser.add_argument("--force", action="store_true", help="ignore AMCL/spread checks")
    parser.add_argument(
        "--navigate",
        action="store_true",
        help="try move_base to configured pose before each record",
    )
    parser.add_argument(
        "--auto-init",
        action="store_true",
        help="publish /initialpose before wizard (uses parking pose)",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="only publish /initialpose and exit",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.10)
    parser.add_argument("--max-spread", type=float, default=0.03)
    parser.add_argument("--nav-timeout", type=float, default=120.0)
    args = parser.parse_args()

    rospy.init_node("auto_calibrate_zones", anonymous=True)
    config = load_mission_config(CONFIG_PATH)
    listener = tf.TransformListener()

    if args.init_only or args.auto_init or config.get("localization", {}).get(
        "auto_initial_pose", False
    ):
        if not apply_auto_initial_pose(config, listener=listener):
            if args.init_only:
                print("无法解析初定位：请先标定 parking 或配置 localization.initial_pose")
                return 1
            print("提示: 未应用自动初定位（parking 未标定时可忽略，用 RViz 设一次）")

    if args.init_only:
        return 0

    steps = _parse_zone_steps(args.zones)
    path = calibration_path(config, SCRIPT_DIR)
    overlay = load_yaml(path)

    print("标定向导开始，共 %d 个区域。Ctrl+C 可随时中断。" % len(steps))
    if not args.write:
        print("当前为 dry-run，结果不会写入文件；确认无误后加 --write")

    try:
        records = run_wizard(
            config,
            overlay,
            steps,
            listener,
            navigate=args.navigate,
            samples=args.samples,
            interval=args.interval,
            max_spread=args.max_spread,
            force=args.force,
            nav_timeout=args.nav_timeout,
        )
    except KeyboardInterrupt:
        print("\n用户中断")
        return 130

    print("\n========== 标定摘要 ==========")
    for label, pose_data, result in records:
        print(
            "%-10s  x=%.3f y=%.3f yaw=%.3f  spread=%.3f"
            % (
                label,
                pose_data["x"],
                pose_data["y"],
                pose_data["yaw"],
                result["spread"],
            )
        )

    if args.write and records:
        save_yaml(path, overlay)
        print("已保存:", path)
        print("建议: python3 preflight_zone_test.py")
    elif records:
        print("dry-run 完成；加 --write 写入 %s" % path)
    else:
        print("未记录任何区域")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

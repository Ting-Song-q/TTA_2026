#!/usr/bin/python3
# coding=UTF-8

import argparse
from pathlib import Path

import rospy

from calibration_utils import (
    apply_auto_initial_pose,
    calibration_path,
    record_zone_pose,
    update_zone_overlay,
)
from config_loader import load_mission_config, load_yaml, save_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "mission_config.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="Record a stable map->base_link pose as a zone center."
    )
    parser.add_argument("zone", choices=("parking", "pickup", "loading", "rescue"))
    parser.add_argument("--zone-id", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.10)
    parser.add_argument("--max-spread", type=float, default=0.03)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-sync-bounds",
        action="store_true",
        help="do not mirror pose into zone_bounds.center",
    )
    parser.add_argument(
        "--auto-init",
        action="store_true",
        help="publish /initialpose from localization config before sampling",
    )
    args = parser.parse_args()
    if args.zone == "rescue" and args.zone_id is None:
        parser.error("--zone-id is required for rescue")

    rospy.init_node("record_zone_pose", anonymous=True)
    config = load_mission_config(CONFIG_PATH)
    if args.auto_init or config.get("localization", {}).get("auto_initial_pose", False):
        apply_auto_initial_pose(config)

    result = record_zone_pose(
        config,
        args.zone,
        zone_id=args.zone_id,
        samples=args.samples,
        interval=args.interval,
        max_spread=args.max_spread,
        force=args.force,
        sync_bounds=not args.no_sync_bounds,
    )
    pose = result["pose"]

    path = calibration_path(config, SCRIPT_DIR)
    overlay = load_yaml(path)
    pose_data = update_zone_overlay(
        overlay,
        args.zone,
        args.zone_id,
        pose,
        sync_bounds=not args.no_sync_bounds,
    )
    print(
        "zone=%s zone_id=%s pose=%s spread=%.3f position_std=%.3f yaw_std=%.3f"
        % (
            args.zone,
            args.zone_id,
            pose_data,
            result["spread"],
            result["position_std"],
            result["yaw_std"],
        )
    )
    if args.write:
        save_yaml(path, overlay)
        print("saved:", path)
    else:
        print("dry run; add --write to save")


if __name__ == "__main__":
    main()

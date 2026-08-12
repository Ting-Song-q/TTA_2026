#!/usr/bin/python3
# coding=UTF-8
"""Enable or disable the calibrated zone-boundary vision path."""

import argparse
from pathlib import Path

from config_loader import load_mission_config, load_yaml, save_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "mission_config.yaml"


def main():
    parser = argparse.ArgumentParser(description="Toggle zone vision safely")
    parser.add_argument("state", choices=("enable", "disable", "status"))
    args = parser.parse_args()

    config = load_mission_config(CONFIG_PATH)
    calibration_path = Path(
        config.get("zone_entry", {}).get(
            "calibration_file", "zone_calibration.yaml"
        )
    )
    if not calibration_path.is_absolute():
        calibration_path = SCRIPT_DIR / calibration_path
    overlay = load_yaml(calibration_path)
    homography = overlay.setdefault("zone_entry", {}).setdefault("homography", {})

    if args.state == "status":
        print("enabled:", bool(homography.get("enabled", False)))
        print("calibration:", calibration_path)
        print("validation_error:", homography.get("validation_error"))
        return

    if args.state == "enable":
        matrix = homography.get("matrix")
        validation_error = homography.get("validation_error")
        if not matrix or validation_error is None:
            raise RuntimeError(
                "cannot enable: calibrated matrix and independent validation are required"
            )
        homography["enabled"] = True
    else:
        homography["enabled"] = False

    save_yaml(calibration_path, overlay)
    print("zone vision %sd: %s" % (args.state, calibration_path))


if __name__ == "__main__":
    main()

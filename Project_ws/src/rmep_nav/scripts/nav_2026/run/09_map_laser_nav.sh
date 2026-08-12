#!/usr/bin/env bash
# 基于雷达地图的小车运动 + 避障测试
#
# 前置:
#   终端1: bash 01_roscore.sh
#   终端2: bash 02_rmep_base.sh
#   终端3: bash 03_map_amcl_move.sh
#   终端4: bash 04_rviz_amcl.sh   # 设 2D Pose Estimate
#   终端5: bash 09_map_laser_nav.sh [--rescue 2]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./_env.sh

echo "[run] map+laser closed-loop nav test"
cd "$NAV_2026_DIR/test"
exec python3 run_map_laser_nav.py "$@"

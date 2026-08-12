#!/usr/bin/env bash
# 终端 10 — 救援任务（2026 任务 + MapLaserNav 动态避障，可选 move_base）
#
# 依赖栈（请先启动）：
#   终端 1: bash 01_roscore.sh
#   终端 2: bash 02_rmep_base.sh
#   终端 3: bash 03_map_amcl_move.sh    # /map=changd.yaml + AMCL（+ EKF；move_base 仅 --nav-mode move_base 需要）
#   终端 4: bash 04_rviz_amcl.sh        # 可选；请先 2D Pose Estimate
#   本终端: bash 10_nav_rescue_closed_loop.sh [--skip-drone --rescue 2]
#           回退旧栈: bash 10_nav_rescue_closed_loop.sh --nav-mode move_base

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./_env.sh

echo "[run] 启动 nav_rescue_2026_closed_loop（默认 MapLaserNav 动态避障）..."
echo "[run] 请确认已先启动 map_amcl_move.launch，并在 RViz 给出 AMCL 初值"
cd "$NAV_2026_DIR"
exec python3 nav_rescue_2026_closed_loop.py "$@"

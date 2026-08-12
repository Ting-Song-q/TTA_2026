#!/usr/bin/env bash
# 救援任务（仅开环 + 方案二白名单避障，参数与 run_car_duo 统一）
#
# 启动顺序（不需要 AMCL / move_base）:
#   终端1: bash 01_roscore.sh
#   终端2: bash 02_rmep_base.sh
#   终端3: bash 08_rescue_2026.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./_env.sh

echo "[run] nav_rescue_2026 (open_loop + zone whitelist)"
cd "$NAV_2026_DIR"
exec python3 nav_rescue_2026.py

#!/usr/bin/env bash
# 终端 6 — 纯小车测试（已改为地图+激光闭环；旧 run_car_only_test 已移除）
#
# 前置: 01 roscore / 02 rmep_base / 03 map_amcl_move / RViz 设初值
# 无地图开环请改用: python3 ../run_car_duo.py --no-init

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./_env.sh

echo "[run] 启动地图+激光闭环小车测试..."
cd "$NAV_2026_DIR/test"
exec python3 run_map_laser_nav.py "$@"

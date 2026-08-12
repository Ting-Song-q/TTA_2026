#!/usr/bin/env bash
# 终端 3 — 地图 + AMCL 定位 + move_base 导航

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./_env.sh

echo "[run] 启动 map_amcl_move（需 roscore 已运行；默认地图 changd.yaml）..."
exec roslaunch rmep_nav map_amcl_move.launch map:=changd.yaml

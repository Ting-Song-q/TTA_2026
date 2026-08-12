#!/usr/bin/env bash
# 终端 2 — 底盘、相机、激光雷达

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./_env.sh

echo "[run] 启动 rmep_base（需 roscore 已运行）..."
exec roslaunch rmep_base rmep_base.launch

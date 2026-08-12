#!/usr/bin/env bash
# 终端 7 — 半自动区域标定向导（可选）

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source ./run/_env.sh

echo "[run] 启动区域标定向导（需 roscore + rmep_base + map_amcl_move 已运行）"
echo "[run] 示例:"
echo "  python3 auto_calibrate_zones.py --write"
echo "  python3 auto_calibrate_zones.py --navigate --write"
echo "  python3 auto_calibrate_zones.py --init-only"
exec python3 auto_calibrate_zones.py "$@"

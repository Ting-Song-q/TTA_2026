#!/usr/bin/env bash
# 终端 5 — 救援任务（仅开环 + 白名单避障，无需 AMCL / move_base）

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./_env.sh

echo "[run] 启动 nav_rescue_2026（open_loop，请确认底盘 /scan 已就绪）..."
cd "$NAV_2026_DIR"
exec python3 nav_rescue_2026.py

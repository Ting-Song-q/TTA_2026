#!/usr/bin/env bash
# 终端 1 — roscore

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "[run] 启动 roscore ..."
exec roscore

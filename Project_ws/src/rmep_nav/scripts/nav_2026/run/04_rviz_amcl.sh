#!/usr/bin/env bash
# 终端 4 — RViz 中手动设初始位姿（2D Pose Estimate），确认定位正常

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./_env.sh

if [[ -z "${DISPLAY:-}" ]]; then
    echo "[run] 当前 SSH 会话没有图形 DISPLAY，不能直接打开 RViz。" >&2
    echo "[run] 请使用带 X11 的 MobaXterm，或在车载 Ubuntu 桌面终端运行本脚本。" >&2
    exit 2
fi

_rviz_cfg="$(rospack find rmep_nav)/param/prams_rviz.rviz"
echo "[run] 启动 RViz: $_rviz_cfg"
echo "[run] 请在 RViz 中使用「2D Pose Estimate」设置初始位姿，确认 AMCL 定位正常后再跑任务。"
exec rviz -d "$_rviz_cfg"

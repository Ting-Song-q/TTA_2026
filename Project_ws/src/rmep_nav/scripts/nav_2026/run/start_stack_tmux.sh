#!/usr/bin/env bash
# 一键在 tmux 中启动终端 1–4（不自动跑任务脚本，等裁判「开始」后手动执行 05）

set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${RESCUE_TMUX_SESSION:-rescue2026}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "[run] 未安装 tmux。请分别开 4 个终端执行:" >&2
    echo "  $RUN_DIR/01_roscore.sh" >&2
    echo "  $RUN_DIR/02_rmep_base.sh" >&2
    echo "  $RUN_DIR/03_map_amcl_move.sh" >&2
    echo "  $RUN_DIR/04_rviz_amcl.sh" >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[run] tmux 会话已存在: $SESSION"
    echo "[run] 附加: tmux attach -t $SESSION"
    exit 1
fi

_new_win() {
    local name="$1"
    local script="$2"
    tmux new-window -t "$SESSION" -n "$name" "bash -lc 'cd \"$RUN_DIR\" && bash \"$script\"; echo; echo \"[$name 已退出] 按 Enter 关闭\"; read'"
}

tmux new-session -d -s "$SESSION" -n roscore \
    "bash -lc 'cd \"$RUN_DIR\" && bash 01_roscore.sh; echo; echo \"[roscore 已退出] 按 Enter 关闭\"; read'"

sleep 1
_new_win base 02_rmep_base.sh
_new_win nav   03_map_amcl_move.sh
if [[ -n "${DISPLAY:-}" ]]; then
    _new_win rviz 04_rviz_amcl.sh
    RVIZ_STATUS="rviz"
else
    RVIZ_STATUS="rviz 未启动（当前会话没有 DISPLAY）"
fi

tmux select-window -t "$SESSION:0"

echo "[run] 已在 tmux 会话 $SESSION 中启动 roscore / rmep_base / map_amcl_move / $RVIZ_STATUS"
echo "[run] 附加会话: tmux attach -t $SESSION"
echo "[run] RViz 中设好初始位姿、裁判「开始」后，另开终端执行:"
echo "       $RUN_DIR/05_nav_rescue.sh"
echo "[run] 停止全部: $RUN_DIR/stop_stack_tmux.sh"

if [[ -t 1 ]] && [[ -z "${RESCUE_TMUX_NO_ATTACH:-}" ]]; then
    exec tmux attach -t "$SESSION"
fi

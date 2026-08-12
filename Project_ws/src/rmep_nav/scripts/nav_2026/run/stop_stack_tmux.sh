#!/usr/bin/env bash
# 停止 start_stack_tmux.sh 创建的 tmux 会话

set -euo pipefail

SESSION="${RESCUE_TMUX_SESSION:-rescue2026}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "[run] 未安装 tmux" >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "[run] 已停止 tmux 会话: $SESSION"
else
    echo "[run] 未找到 tmux 会话: $SESSION"
fi

#!/usr/bin/env bash
# 公共 ROS 环境：由其它 run/*.sh source，勿直接执行。

_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_2026_DIR="$(cd "$_RUN_DIR/.." && pwd)"
PROJECT_WS="$(cd "$NAV_2026_DIR/../../../.." && pwd)"

_setup_bash="$PROJECT_WS/devel/setup.bash"
if [[ ! -f "$_setup_bash" ]]; then
    echo "[run] 错误: 未找到 $_setup_bash" >&2
    echo "[run] 请先编译: cd $PROJECT_WS && catkin_make" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$_setup_bash"

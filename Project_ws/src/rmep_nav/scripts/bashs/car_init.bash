#!/bin/bash

# 开机自启，sleep 12,用于确保硬件均正确初始化
sleep 12

# 定义输出文件路径
output_file="/tmp/car_init.log"

# 清理旧的输出文件
[ -f "$output_file" ] && rm "$output_file"

# 定义清理函数
cleanup() {
    echo "Cleaning up..."
    rm -f "$output_file"
    kill $(jobs -p) 2>/dev/null
}

# 捕获终止信号并调用清理函数
trap cleanup EXIT

script -q -a -f -c "
    (
        cd /home/tta/Project_ws
        source ./devel/setup.bash
        roslaunch rmep_base rmep_base.launch
    ) &

    sleep 12

    (
        cd /home/tta/Project_ws
        source ./devel/setup.bash
        roslaunch rmep_nav map_amcl_move.launch
    )
" $output_file &
script_pid=$!

# 输出提示
echo "All commands executed. Logs are being saved to $output_file"

# 等待 script 进程完成
wait $script_pid

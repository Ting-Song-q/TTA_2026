#!/bin/bash
# 初始延迟，用于确保硬件正确初始化
sleep 210

# 定义输出文件路径
output_file="/tmp/car_auto.log"

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

# 使用 script 记录所有命令的输出
script -q -a -f -c "
    (
        # 第一个命令
        cd /home/tta/Project_ws
        source ./devel/setup.bash
        echo 'Starting server2.py...'
        rosrun rmep_nav server2.py &
        server2_pid=\$!
        
        # 第二个延迟
        sleep 5
        
        # 第二个命令
        echo 'Starting nav_2024_static_changed2.py...'
        rosrun rmep_nav nav_2024_static_changed2.py &
        nav2_pid=\$!

        # 等待所有后台进程完成
        wait \$server2_pid
        wait \$nav2_pid
    )
" $output_file &

# 等待 script 进程完成
wait

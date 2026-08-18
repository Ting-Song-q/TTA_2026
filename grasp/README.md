# Mobile Grasp Prototype

This directory is independent from `arm/` and contains the vehicle-assisted grasp prototype.

```text
arm grasp pose
-> vehicle forward 0.10 m
-> vehicle-only lateral visual alignment (block center to image x=0.55)
-> vehicle forward 0.10 m
-> wrist-camera visual grasp
-> vehicle backward by the commanded entry distance
-> arm final pose
```

The ROS base uses `/cmd_vel`: `linear.x` is forward/backward and `linear.y` is left/right.
`base_alignment` uses fixed `lateral_speed_mps` and linearly interpolates each correction pulse from `min_lateral_duration_s` to `max_lateral_duration_s`. Errors within `tolerance_px` finish alignment; the interpolation range is from `tolerance_px` to `max_error_px`. `lateral_direction_sign` controls the physical direction. Each pulse completes and stops before the next camera frame is captured.
When several red regions pass the detector filters, lateral alignment selects the candidate whose center is closest to the `target_u_ratio` vertical line. Debug images show every candidate with a thin cyan box and the selected target with the existing yellow contour, red box, and magenta cross.

## 启动流程

确认机械臂、腕部相机和底盘均已接通后，按以下顺序打开三个终端。

终端 1：启动 ROS 主节点。

```bash
roscore
```

终端 2：启动底盘驱动。

```bash
source ~/Project_ws/devel/setup.bash
roslaunch rmep_base rmep_base.launch
```

终端 3：加载工作空间、进入 `lerobot` 环境并启动抓取原型。先做检查，确认后再执行真实运动。

```bash
source ~/Project_ws/devel/setup.bash
conda activate lerobot
cd ~/tta
/usr/bin/python3 -c "import rospy, rospkg; print('system ROS Python OK')"
python3 -u grasp/mobile_grasp.py --config grasp/configs/mobile_grasp.yaml --yes
```

若脚本停在 ROS 初始化前后，先在终端 3 检查 ROS Master：

```bash
echo $ROS_MASTER_URI
rosnode list
```

`rosnode list` 必须能返回节点列表；否则先检查终端 1 的 `roscore` 是否仍在运行。

若 `rosnode list` 正常、但脚本停在 `initialize ROS node`，说明 Conda Python 注册自身节点时的主机名解析异常。底盘与脚本都在同一台 EVA 主机时，在终端 3 执行：

```bash
unset ROS_HOSTNAME
export ROS_IP=127.0.0.1
timeout 10s python3 -u -c "import rospy; print('before init', flush=True); rospy.init_node('mobile_grasp_probe', anonymous=True); print('ROS init OK', flush=True)"
```

看到 `ROS init OK` 后，再用同一终端运行原型脚本。若该命令仍超时，记录 `env | grep '^ROS'` 与 `hostname -I` 的输出后再检查网络配置。

`lerobot` 环境必须包含 ROS Python 依赖。若出现 `No module named 'rospkg'`，在该环境执行：

```bash
python3 -m pip install rospkg
```

Calibrate the base at low speed first. `approach_before_align_m`, `approach_after_align_m` and the lateral gain are open-loop values. The script sends a zero `Twist` on every exit path, then retreats by the distance it commanded before returning the arm to `poses.final`.


conda activate lerobot
cd /home/tta/tta
python3 arm/arm.py \
  --raw \
  --read \
  --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61032896-if00

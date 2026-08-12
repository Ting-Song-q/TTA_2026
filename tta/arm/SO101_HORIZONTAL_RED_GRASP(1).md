# SO-101 红色方块水平抓取调试

脚本：[so101_horizontal_red_grasp.py](so101_horizontal_red_grasp.py)。

这是一套不需要训练数据的视觉伺服原型。它假设小车已经停在货架外的固定参考位置，SO-101 先到固定观察位，随后腕部相机检测红色 `20 x 20 x 20 mm` 方块，再用小范围关节增量修正位置。

## 安全规则

- 真实运行前完成 LeRobot 的 SO-101 标定，且机械臂周围没有人、线缆或货架障碍。
- 第一次只运行 `--observe-only`；没有确认观察位安全前，不要运行完整抓取。
- 任何异常立即拔掉机械臂电源或按 `Ctrl+C`。`Ctrl+C` 只能停止脚本，不能替代断电急停。
- 当前版本没有货架三维模型、力反馈或自动碰撞检测；所有位姿与增量必须低速逐步验证。

## 文件和输出

所有可调参数在 [horizontal_red_grasp_config.yaml](horizontal_red_grasp_config.yaml)。

相机测试图和对准过程图默认写入：

```text
F:\car_2026\tta\output\so101_red_block_camera\
```

完整抓取脚本每次视觉对准会保存 `align_000.jpg`、`align_001.jpg` 等标注图；黄色十字是 `desired_center`，红框是检测到的方块。

## 第一次配置

1. 确认腕部相机编号，例如 `1`。使用 `so101_red_block_camera_test.py --camera 1` 保存图片确认。
2. 确认 SO-101 从动臂的串口，例如 Windows 的 `COM5`，或 Linux 的 `/dev/ttyACM0`。
3. 在 YAML 中把 `grasp.motion_enabled` 改为 `true`。
4. 手动/遥操作将机械臂移动到一个安全观察姿态：夹爪在货架外、打开后不会碰边框、相机能清楚看到方块。记录每个关节的 LeRobot 角度，填入 `grasp.observe_pose`。
5. 确认 `gripper_open` 和 `gripper_close` 的方向。SO-101 的夹爪通常使用 `0-100` 范围；不要假设占位的 `1.0` 和 `0.0` 与实际一致。

## 只调观察位

此命令只连接、移动到 `observe_pose`、打开夹爪，然后断开。不会拍照、视觉微调、前伸或闭夹爪。

```powershell
conda activate arm
cd F:\car_2026\tta
python test\so101_horizontal_red_grasp.py --observe-only --port COM5 --yes
```

若姿态不安全，先断电或 `Ctrl+C`，然后修改 YAML 中的 `grasp.observe_pose`。重复此步骤直到相机与夹爪位置合适。

## 只验证视觉微调

先将以下三段全部改为 `0.0`：

```yaml
grasp:
  pregrasp_delta: {shoulder_pan: 0.0, shoulder_lift: 0.0, elbow_flex: 0.0, wrist_flex: 0.0, wrist_roll: 0.0}
  insert_delta: {shoulder_pan: 0.0, shoulder_lift: 0.0, elbow_flex: 0.0, wrist_flex: 0.0, wrist_roll: 0.0}
  retreat_delta: {shoulder_pan: 0.0, shoulder_lift: 0.0, elbow_flex: 0.0, wrist_flex: 0.0, wrist_roll: 0.0}
```

然后运行完整流程：

```powershell
python test\so101_horizontal_red_grasp.py --camera 1 --port COM5 --execute --yes
```

观察控制方向：方块在图像左侧时，执行一次后应更靠近黄色十字。如果反向，翻转 `pan_deg_per_pixel` 的符号；前后方向反向时，同时翻转 `reach_deg_per_pixel` 的符号。每次只改一个参数，再重新测试。

## 加入抓取动作

视觉微调正确后，按照很小增量依次配置：

1. `pregrasp_delta`：保持夹爪在方块前方的准备姿态。
2. `insert_delta`：最后短距离近水平插入。先只设置 `elbow_flex`，每次改动不超过 1-2 度，确认不会碰货架底板或侧框后再加入肩部/腕部补偿。
3. `gripper_close`：闭爪值。
4. `retreat_delta`：应与插入方向相反，并略微抬升。

建议最终保持如下顺序：观察位 -> 对准 -> 预抓取 -> 短距离插入 -> 闭爪 -> 后退。不要让视觉微调阶段直接深入货架。

## YAML 参数速查

| 参数 | 用途 |
| --- | --- |
| `camera.index_or_path` | 腕部相机编号，例如 `1`，或 Linux 的 `/dev/video_wrist`。 |
| `detector.*` | HSV/RGB 红色筛选、ROI 和面积阈值。画面识别不稳时调这里。 |
| `motion_enabled` | 真实运动总开关。`false` 时任何运动命令都会被拒绝。 |
| `observe_pose` | 已验证的安全观察关节姿态。执行开始时首先到达这里。 |
| `desired_center` | 黄色十字位置，不必是图像中心；应标定为方块被夹爪中心线对准时所在像素。 |
| `pixel_deadband` | 允许的像素误差。先设大一些，例如 `15-20`，稳定后再减小。 |
| `pan_deg_per_pixel` | 图像水平误差转底座关节的小步增益；若左右修正方向相反则改符号。 |
| `reach_deg_per_pixel` | 图像垂直误差转肩/肘联动的小步增益；若前后修正方向相反则改符号。 |
| `max_joint_step_deg` | 单次视觉微调的最大关节变化，首次建议 `1.0-2.0`。 |
| `pregrasp_delta` | 对准后到物块前方的关节增量。 |
| `insert_delta` | 最后短距离插入物块的关节增量。 |
| `retreat_delta` | 闭爪后收回的关节增量。 |

## 目前的限制

当前控制器是关节空间的经验增量控制。LeRobot 支持使用 SO-101 的 URDF 将末端执行器的绝对/相对位姿转换为关节目标；在观察位、目标中心和插入距离完成实机标定后，建议下一版换成末端 `mm` 级增量和逆运动学。这样“向左 3 mm、向前 4 mm”的含义更稳定，也更易加工作空间约束。

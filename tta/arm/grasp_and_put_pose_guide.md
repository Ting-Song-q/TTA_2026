# SO101 抓取与投放：四个锚点姿态

当前版本只需要标定四个姿态。目标物块和红色投放框的具体位置由腕部相机识别，再通过手眼标定、像素到基座坐标转换和 SO101 IK 自动生成路径。

## 必须标定的四个姿态

| 姿态 | 作用 | 夹爪 |
|---|---|---|
| `grasp_observe` | 能看到飞机载物盒和红色物块的观察位 | 张开 |
| `grasp_reset` | 抓取成功后抬起并离开盒子的安全携带位 | 闭合 |
| `place_observe` | 抓住物块后能看到外部红色空心框的观察位 | 闭合 |
| `safe` | 投放结束后的安全复位位 | 张开 |

YAML 中的四组 ticks 目前仍是占位值，需要在真实 SO101 上重新示教。`initial` 已经取消，不再作为额外姿态；程序启动后直接移动到 `grasp_observe`。

## 哪些动作是自动规划的

从 `grasp_observe` 开始，程序会识别红色物块中心像素，使用腕部相机手眼标定将像素转换为机械臂基座坐标，在目标点上方增加 `approach_clearance_m`，用 IK 规划到达，再继续规划到抓取高度完成近竖直下降，闭合夹爪后自动回到 `grasp_reset`。

抓取后的顺序是：

```text
grasp_observe
  -> IK 自动到物块上方
  -> IK 自动竖直下降
  -> 夹爪闭合
  -> IK/锚点复位到 grasp_reset
  -> place_observe
```

到达 `place_observe` 后，程序重新使用同一个腕部相机识别红色空心框，将红框中心转换为基座坐标，IK 自动规划到红框上方并下降到投放高度，张开夹爪，最后回到 `safe`。

## 四个姿态的标定要求

### `grasp_observe`

相机必须同时看到载物盒和红色物块，夹爪应位于可规划工作空间内。这个姿态决定抓取视觉坐标的初始参考。

### `grasp_reset`

夹住物块后，机械臂必须能从盒子中抬出物块并避开盒边。它是抓取成功后的携带复位位，不负责决定物块的具体抓取位置。

### `place_observe`

夹爪保持闭合，腕部相机应能看到外部红色空心框。物块不能完全遮挡红框，机械臂还要留出 IK 可达空间。

### `safe`

释放物块后回到远离红框和障碍物的位置，夹爪保持张开。

## 仍需配置的非姿态参数

```yaml
intrinsics: output/camera_calib/camera_intrinsics.yaml
handeye: output/handeye_ee_cam.yaml
scene:
  table_z_m: ...
  place_z_m: ...
motion:
  approach_clearance_m: 0.06
```

`table_z_m` 要对应飞机载物盒中物块所在的参考平面，`place_z_m` 要对应红框内部实际放置平面。二者不是相机观察高度。

## 与之前版本的区别

之前版本把多个 approach/down/lift/carry 姿态都作为固定姿态，这是过度写死的。当前版本已经删除这些姿态：它们变成由像素坐标、标定参数和 IK 在运行时生成的路径。

只有四个观察/复位锚点保留为人工标定值。

## 开启实机动作前

1. 将四个姿态替换为现场实测 ticks；
2. 确认 `intrinsics` 和 `handeye` 与腕部相机对应；
3. 先分别测试抓取链和投放链；
4. 确认 `table_z_m`、`place_z_m` 和 `approach_clearance_m`；
5. 最后将 `motion_enabled` 改为 `true` 并使用 `--yes`。

## 实际启动命令

在项目根目录 `/home/tta/tta` 下执行。

先做不连接机械臂的启动检查：

```bash
cd /home/tta/tta
python3 arm/grasp_and_put.py \
  --config arm/grasp_and_put.yaml \
  --dry-run
```

确认四个姿态、相机、标定文件和工作空间都已经实测后，将配置中的：

```yaml
motion_enabled: true
```

然后执行完整的真实流程：

```bash
cd /home/tta/tta
python3 arm/grasp_and_put.py \
  --config arm/grasp_and_put.yaml \
  --yes
```

真实流程会依次完成：

```text
grasp_observe
-> 自动识别并 IK 抓取
-> grasp_reset
-> place_observe
-> 自动识别并 IK 投放
-> safe
```

当前脚本没有单独的“只测试抓取段”或“只测试投放段”命令；首次实机测试应在旁边有人急停/断电的情况下运行完整流程。

## 使用 `arm.py` 读取标定位姿

可以用 `arm.py` 读取 SO101 当前的原始关节 ticks，用于标定四个锚点姿态。将机械臂手动摆到目标姿态后，在项目根目录执行：

```bash
cd /home/tta/tta
python3 arm/arm.py \
  --raw \
  --read \
  --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61032896-if00
```

如果当前终端已经位于 `arm/` 目录，则命令写成：

```bash
python3 arm.py \
  --raw \
  --read \
  --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61032896-if00
```

终端会打印：

```text
Current raw encoder ticks:
  shoulder_pan: ...
  shoulder_lift: ...
  elbow_flex: ...
  wrist_flex: ...
  wrist_roll: ...
  gripper: ...
```

分别把四个现场读取结果填入 `grasp_and_put.yaml` 的：

```yaml
poses:
  grasp_observe: ...
  grasp_reset: ...
  place_observe: ...
  safe: ...
```

`--raw --read` 只读取当前位置，不发送运动指令。标定时应依次手动摆到四个姿态并记录 ticks；不要把 `arm.py` 中其他项目的 `observe_pose` 或 `raw_observe_pose` 直接当成这四个姿态使用。

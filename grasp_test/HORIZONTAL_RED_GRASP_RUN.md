# ArmPi Pro Forward Grasp Runbook

Applies to:
- [horizontal_red_grasp_test.py](/abs/path/F:/car_2026/Project_ws/src/rmep_nav/scripts/test/horizontal_red_grasp_test.py)

This is the only script you need to keep for the current ArmPi Pro car.

It is for:
- forward-facing grasp
- arm-first visual calibration
- red target center marking
- debug image output during calibration and grasp

It is not for:
- top-down grasp
- RoboMaster EP
- `rmep_base`
- `/ep_arm` or `/ep_gripper`

It now uses the current ArmPi Pro interfaces:
- `/arm_controller/command`
- `/gripper_controller/command`
- `/joint_states`
- `/usb_cam/image_raw/compressed`

## 1. What This Script Does

The full flow is:
1. move the arm to the forward observe pose
2. wait for a fresh camera frame
3. detect the largest red target in the configured ROI
4. save debug images with:
   - ROI
   - target contour
   - target bbox
   - detected center
   - desired center
   - `err_u / err_v`
   - mask preview
5. do arm-only visual alignment
6. execute a forward grasp sequence:
   - open gripper
   - move home
   - move pregrasp
   - move grasp
   - close gripper
   - retreat

So this script already matches what you want:
- first calibrate position with the arm
- then do a forward grasp
- not a top-down grasp

## 2. Why The Other Script Was Removed

`armpi_grasp_action_test.py` was only a minimal smoke test:
- useful for first interface bring-up
- but redundant after the ArmPi Pro visual grasp flow was adapted

`horizontal_red_grasp_test.py` is the better one to keep because it already includes:
- observe pose
- calibration images
- target center marking
- forward grasp sequence

## 3. Required Topics

Make sure these exist:

```bash
/arm_controller/command
/gripper_controller/command
/joint_states
/usb_cam/image_raw/compressed
```

Quick check:

```bash
rostopic list | grep -E "arm_controller|gripper_controller|joint_states|usb_cam"
```

## 4. Startup

Terminal 1:

```bash
roscore
```

Terminal 2:

```bash
cd ~/Desktop/Project_ws
source /opt/ros/melodic/setup.bash
```

If your ArmPi Pro factory stack is not already running, start it first.

## 5. Run

Dry run first:

```bash
python3 src/rmep_nav/scripts/test/horizontal_red_grasp_test.py --dry-run
```

`--dry-run` means:
- no arm motion is published
- no gripper motion is published
- camera capture still runs
- red target detection still runs
- debug images still save

So yes: in dry run, the arm should not move.

Real execution:

```bash
python3 src/rmep_nav/scripts/test/horizontal_red_grasp_test.py
```

## 6. Forward Grasp Poses

The script currently uses these built-in forward grasp poses:

```text
observe  = [0.0, 0.524, -1.361, -1.759, 0.0]
home     = [0.0, 0.524, -1.361, -1.759, 0.0]
pregrasp = [0.0, 0.604, -1.301, -1.819, 0.0]
grasp    = [0.0, 0.644, -1.271, -1.849, 0.0]
retreat  = [0.0, 0.724, -1.301, -1.819, 0.0]
```

These are forward-grasp starter values for ArmPi Pro.
They are meant to be calibrated on the real car.

## 7. Calibration Images

Debug output directory:

```bash
src/rmep_nav/scripts/test/debug/horizontal_red_grasp/
```

Typical files:
- `000_detect.jpg`
- `001_arm_align_01.jpg`
- `002_arm_align_02.jpg`
- `003_aligned.jpg`
- `004_after_grasp.jpg`

These images are the calibration trace.
Use them to judge:
- whether the target center is detected correctly
- whether the desired center is placed correctly
- whether arm-only alignment is moving in the right direction
- whether the grasp pose is too short or too long

## 8. Useful Logs

- `move arm to observe pose: ...`
  The script is moving to the forward observe pose.

- `waiting for fresh camera stream...`
  It is waiting for `/usb_cam/image_raw/compressed`.

- `saved debug image: ...`
  A calibration/debug image was saved.

- `arm-align X: err_u=... err_v=...`
  The script is calibrating the forward grasp with arm-only visual alignment.

- `arm-only aligned: ...`
  The visual calibration reached the deadband.

- `horizontal red grasp flow completed`
  The whole forward grasp sequence completed.

## 9. Configuration

You can optionally pass a YAML file:

```bash
python3 src/rmep_nav/scripts/test/horizontal_red_grasp_test.py --config /path/to/config.yaml
```

Useful values to tune in config:
- `vision.detector.roi`
- `vision.detector.min_area`
- `vision.detector.min_rect_fill`
- `vision.grasp.align.center_offset`
- `vision.grasp.align.pixel_deadband`
- `vision.grasp.align.arm_gain_u`
- `vision.grasp.align.arm_gain_v`
- `vision.grasp.align.arm_step_limit`
- `vision.grasp.arm.observe`
- `vision.grasp.arm.pregrasp`
- `vision.grasp.arm.grasp`
- `vision.grasp.arm.retreat`

## 10. Minimal Commands

Dry run:

```bash
cd ~/Desktop/Project_ws
source /opt/ros/melodic/setup.bash
python3 src/rmep_nav/scripts/test/horizontal_red_grasp_test.py --dry-run
```

Real run:

```bash
cd ~/Desktop/Project_ws
source /opt/ros/melodic/setup.bash
python3 src/rmep_nav/scripts/test/horizontal_red_grasp_test.py
```

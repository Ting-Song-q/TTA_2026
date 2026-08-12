#!/usr/bin/python3
# coding=utf8
import sys
import cv2
import math
import time
import rospy
import numpy as np
from threading import RLock, Thread

from std_srvs.srv import *
from std_msgs.msg import *
from sensor_msgs.msg import Image
from sensor.msg import Led
from hiwonder_servo_msgs.msg import MultiRawIdPosDur
from chassis_control.msg import SetTranslation, SetVelocity

# Import project libraries
from armpi_pro import PID
from armpi_pro import Misc
from armpi_pro import bus_servo_control
from armpi_pro import apriltag
from kinematics import ik_transform

class IntegratedTransportTask:
    def __init__(self):
        rospy.init_node('integrated_transport_task', log_level=rospy.INFO)
        
        self.lock = RLock()
        self.ik = ik_transform.ArmIK()
        
        # Publishers
        self.joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        self.set_velocity = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)
        self.buzzer_pub = rospy.Publisher('/sensor/buzzer', Float32, queue_size=1)
        self.rgb_pub = rospy.Publisher('/sensor/rgb_led', Led, queue_size=1)
        self.image_pub = rospy.Publisher('/visual_processing/image_result', Image, queue_size=1)
        
        # Parameters
        self.color_range_list = rospy.get_param('/lab_config_manager/color_range_list', {})
        self.img_w = 640
        self.img_h = 480
        self.size_m = (320, 240)
        
        # AprilTag Detector
        # Specify tag25h9 family explicitly
        # Note: The custom 'armpi_pro.apriltag' wrapper does NOT support 'families' argument in constructor.
        # We must initialize it without arguments (except searchpath) and rely on its default options.
        # It seems we cannot easily change the family to 25h9 via this wrapper's init.
        # Reverting to default initialization to fix the TypeError crash.
        self.detector = apriltag.Detector(searchpath=apriltag._get_demo_searchpath())
        
        # Detection State
        self.mode = 'None' # 'None', 'color', 'apriltag'
        self.target_color = 'None'
        self.target_tag_id = None
        self.detect_result = None # {'center_x': int, 'center_y': int, 'data': any}
        self.updated = False
        
        # PID Controllers
        self.x_pid = PID.PID(P=0.06, I=0, D=0)
        self.y_pid = PID.PID(P=0.00003, I=0, D=0)
        
        # Camera Subscriber
        # Initialize running state before subscriber
        self.running = True
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        
        # Task State
        self.running = True
        self.centreX = 320
        self.centreY = 410
        self.x_dis = 500
        self.y_dis = 0.15
        
        self.range_rgb = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
        }
        
        rospy.sleep(1.0)
        self.init_robot()
        
        # Start Task Thread
        self.task_thread = Thread(target=self.run_task)
        self.task_thread.setDaemon(True)
        self.task_thread.start()

    def init_robot(self):
        rospy.loginfo("Initializing Robot...")
        self.stop_chassis()
        self.set_rgb('black')
        self.reset_arm()
        
    def reset_arm(self):
        # Move arm to initial driving position
        target = self.ik.setPitchRanges((0, 0.15, 0.10), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, (
                (1, 200), (2, 500), 
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, servo_data['servo6'])
            ))
        rospy.sleep(1.5)
        
    def set_rgb(self, color):
        if color not in self.range_rgb:
            r,g,b = 0,0,0
        else:
            r,g,b = self.range_rgb[color][2], self.range_rgb[color][1], self.range_rgb[color][0]
            
        led = Led()
        led.index = 0
        led.rgb.r = r; led.rgb.g = g; led.rgb.b = b
        self.rgb_pub.publish(led)
        led.index = 1
        self.rgb_pub.publish(led)

    def stop_chassis(self):
        self.set_velocity.publish(0, 90, 0)

    def move_chassis(self, speed, direction, omega):
        self.set_velocity.publish(speed, direction, omega)

    # --- Image Processing ---
    def getAreaMaxContour(self, contours):
        contour_area_temp = 0
        contour_area_max = 0
        area_max_contour = None
        for c in contours:
            contour_area_temp = math.fabs(cv2.contourArea(c))
            if contour_area_temp > contour_area_max:
                contour_area_max = contour_area_temp
                if contour_area_temp > 10:
                    area_max_contour = c
        return area_max_contour, contour_area_max

    def color_detect(self, img, color):
        if color == 'None':
            return img, None
            
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]
        frame_resize = cv2.resize(img_copy, self.size_m, interpolation=cv2.INTER_NEAREST)
        # Use HSV as requested
        frame_hsv = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2HSV)
        
        result = None
        
        # Hardcoded HSV ranges for robustness
        # Format: (Lower H, Lower S, Lower V), (Upper H, Upper S, Upper V)
        # Note: OpenCV H is 0-180, S/V 0-255
        hsv_ranges = {
            'red': [
                ((0, 43, 46), (10, 255, 255)),
                ((156, 43, 46), (180, 255, 255))
            ],
            'green': [((35, 43, 46), (77, 255, 255))],
            'blue': [((100, 43, 46), (124, 255, 255))]
        }

        if color in hsv_ranges:
            mask = np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in hsv_ranges[color]:
                mask |= cv2.inRange(frame_hsv, np.array(lower), np.array(upper))
            
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            area_max_contour, area_max = self.getAreaMaxContour(contours)

            if area_max > 100:
                (centerx, centery), radius = cv2.minEnclosingCircle(area_max_contour)
                cx = int(Misc.map(centerx, 0, self.size_m[0], 0, img_w))
                cy = int(Misc.map(centery, 0, self.size_m[1], 0, img_h))
                rad = int(Misc.map(radius, 0, self.size_m[0], 0, img_w))
                
                cv2.circle(img, (cx, cy), rad+5, self.range_rgb[color], 2)
                
                # Assign ID based on color for compatibility with intelligent_grasp logic
                # Red: 1, Green: 2, Blue: 3
                color_id = 0
                if color == 'red': color_id = 1
                elif color == 'green': color_id = 2
                elif color == 'blue': color_id = 3
                
                result = {'center_x': cx, 'center_y': cy, 'data': color_id}
                
        return img, result

    def apriltag_detect(self, img, target_id):
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]
        frame_resize = cv2.resize(img_copy, self.size_m, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray, return_image=False)
        
        result = None
        id_smallest = None
        
        if len(detections) != 0:
            for i, detection in enumerate(detections):
                tag_id = int(detection.tag_id)
                
                # Filter by ID
                if target_id is not None and tag_id != target_id:
                    continue
                
                corners = np.rint(detection.corners)
                for i in range(4):
                    corners[i][0] = int(Misc.map(corners[i][0], 0, self.size_m[0], 0, img_w))
                    corners[i][1] = int(Misc.map(corners[i][1], 0, self.size_m[1], 0, img_h))
                
                cv2.drawContours(img, [np.array(corners, np.int)], -1, (0, 255, 255), 6)
                object_center_x = int(Misc.map(detection.center[0], 0, self.size_m[0], 0, img_w))
                object_center_y = int(Misc.map(detection.center[1], 0, self.size_m[1], 0, img_h))
                
                cv2.putText(img, str(tag_id), (object_center_x - 10, object_center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 255, 255], 6)
                
                # Pick the smallest ID if multiple found (or just the target one)
                if id_smallest is None or tag_id <= id_smallest:
                    id_smallest = tag_id
                    result = {'center_x': object_center_x, 'center_y': object_center_y, 'data': tag_id}
                    
        return img, result

    def image_callback(self, ros_image):
        if not self.running:
            return
            
        try:
            image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
            cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            rospy.logerr(f"Image decode error: {e}")
            return

        processed_img = cv2_img
        res = None

        with self.lock:
            if self.mode == 'color':
                processed_img, res = self.color_detect(cv2_img, self.target_color)
            elif self.mode == 'apriltag':
                processed_img, res = self.apriltag_detect(cv2_img, self.target_tag_id)
            
            self.detect_result = res
            self.updated = True

        # Publish result image
        try:
            rgb_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB).tobytes()
            ros_image.data = rgb_image
            self.image_pub.publish(ros_image)
        except Exception as e:
            pass

    # --- Task Logic ---
    def move_to_cargo_area(self):
        rospy.loginfo("Moving to Cargo Area...")
        self.move_chassis(50, 90, 0)
        rospy.sleep(3.0)
        self.stop_chassis()
        rospy.sleep(0.5)

    def return_to_parking(self):
        rospy.loginfo("Returning to Parking Area...")
        self.move_chassis(0, 0, 0.5) # Rotate
        rospy.sleep(1.5)
        self.stop_chassis()
        self.move_chassis(50, 90, 0)
        rospy.sleep(3.0)
        self.stop_chassis()

    def search_color(self, color):
        rospy.loginfo(f"Searching for {color} block (Arm Left)...")
        with self.lock:
            self.mode = 'color'
            self.target_color = color
            self.detect_result = None
        
        # Turn Arm Base (Servo 6) to Left (e.g., 800-900 range)
        # Default is 500 (Center). Left is usually > 500 or < 500 depending on orientation.
        # Assuming ID 6: 0-1000 range. 500 Center. 
        # Let's sweep ID 6 from 600 to 900 (Left)
        
        found = False
        start_time = time.time()
        
        # Initial look at Left
        # Dynamic Search: Rotate Arm (Servo 6) and Chassis (Omega)
        # Search range: Servo 6 from 600 to 900, Chassis slowly rotating left
        
        # Start scanning
        scan_positions = [600, 700, 800, 900]
        for pos in scan_positions:
            bus_servo_control.set_servos(self.joints_pub, 1500, ((6, pos),))
            rospy.sleep(0.8)
            
            # Check for color
            with self.lock:
                if self.detect_result:
                    rospy.loginfo(f"Found {color} at arm pos {pos}!")
                    found = True
                    break
            
            # Also rotate chassis slightly if needed (dynamic)
            # self.move_chassis(0, 0, 0.2)
            # rospy.sleep(0.5)
            # self.stop_chassis()
            
        if not found:
             # Try rotating chassis back a bit and scan again?
             pass
            
        return found

    def align_and_grasp(self, color):
        rospy.loginfo(f"Aligning to grasp {color}...")
        self.set_rgb(color)
        
        # NOTE: Chassis is stationary now. We use Arm Base (ID 6) to center the object.
        
        # 1. Arm Grasping Logic
        bus_servo_control.set_servos(self.joints_pub, 800, ((1, 120),)) # Open Claw
        rospy.sleep(0.5)
        
        # Move to Hover Position (Arm Left)
        # We need to maintain the current ID 6 position or search position
        # Let's assume we are roughly at 800.
        
        target = self.ik.setPitchRanges((0, 0.15, 0.03), -180, -180, 0)
        if target:
            servo_data = target[1]
            # Keep ID 6 at current search position or dynamic?
            # We will use PID to drive ID 6.
            bus_servo_control.set_servos(self.joints_pub, 2000, (
                (1, 120), (2, 500), 
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']) # Don't move 6 yet, let PID handle it
            ))
        rospy.sleep(1.5)
        
        self.x_pid.clear()
        self.y_pid.clear()
        # Initialize x_dis to current Servo 6 position (approx 800 for Left)
        self.x_dis = 800 
        self.y_dis = 0.15
        
        stable_count = 0
        
        # Variables for stability check (position_en equivalent)
        last_x = 0
        last_y = 0
        stable_pos_count = 0
        position_en = False
        
        # Variables for final grasp trigger
        stable_track_count = 0
        arm_move = False
        offset_y = 0
        color_buf = []
        
        # Timeout logic for alignment
        align_start_time = time.time()
        
        # Phase 1: Chassis Alignment (Coarse)
        rospy.loginfo("Phase 1: Chassis Alignment...")
        chassis_aligned = False
        
        while not chassis_aligned:
            # Timeout check
            if time.time() - align_start_time > 30:
                rospy.logwarn("Chassis alignment timed out!")
                return False
                
            res = None
            with self.lock:
                res = self.detect_result
                
            if res:
                cx = res['center_x']
                cy = res['center_y'] # Y in image correlates to Distance
                
                # Logic:
                # 1. Strafe Left/Right to adjust distance (cy) - Keeping car facing forward
                # 2. Strafe Forward/Back (relative to car) to center block horizontally (cx)
                
                # Note:
                # Car Orientation: Facing Forward (0 deg)
                # Arm Orientation: Facing Left (90 deg)
                # Camera Orientation: On Arm, facing Left
                
                # Axis Mapping:
                # Camera X (cx): Horizontal in image. 
                # - If block is Left in Image (cx < 320), it is "Forward" relative to Car.
                # - If block is Right in Image (cx > 320), it is "Back" relative to Car.
                # Action: Move Car Forward/Back to center cx.
                
                # Camera Y (cy): Vertical in image (Distance).
                # - If block is Top of Image (cy small), it is "Far" from Arm (Left side of car).
                # - If block is Bottom of Image (cy large), it is "Close" to Arm.
                # Action: Move Car Left/Right (Strafe) to adjust distance.
                
                # Control 1: Centering (cx) -> Car Forward/Back (0 / 180)
                vx = 0
                err_x = self.centreX - cx
                if err_x > 20: # Block is Left in image -> Forward relative to car
                    vx = 30 # Slow speed
                    dir_x = 0
                elif err_x < -20: # Block is Right in image -> Back relative to car
                    vx = 30
                    dir_x = 180
                else:
                    vx = 0
                    dir_x = 0
                    
                # Control 2: Distance (cy) -> Car Left/Right (90 / 270)
                # Target Distance: Increase distance (cy smaller)
                # Old range: 200-350. New target: 150-250 (Further away)
                vy = 0
                if cy < 150: # Too far (Top of image) -> Move Left (Towards block)
                    vy = 30 # Slow speed
                    dir_y = 90
                elif cy > 250: # Too close (Bottom of image) -> Move Right (Away from block)
                    vy = 30
                    dir_y = 270
                else:
                    vy = 0
                    dir_y = 0
                
                # Combine Movements?
                # Simple logic: Prioritize Distance (Y) then Centering (X) or mix?
                # Let's mix vectors roughly or just switch.
                # Since move_chassis takes (speed, direction, omega), we need a resultant vector.
                # But simple strafing is safer.
                
                if vy != 0:
                    self.move_chassis(vy, dir_y, 0) # Strafe Left/Right
                elif vx != 0:
                    self.move_chassis(vx, dir_x, 0) # Move Fwd/Back
                else:
                    rospy.loginfo("Chassis Aligned! Switching to Arm Control.")
                    self.stop_chassis()
                    chassis_aligned = True
                    
                # No rotation (omega=0) to keep body parallel
                
            else:
                self.stop_chassis()
                rospy.sleep(0.05)
            
            rospy.sleep(0.05)
            
        # Phase 2: Arm Fine-tuning & Grasp
        rospy.loginfo("Phase 2: Arm Fine-tuning...")
        
        while not arm_move:
            # Timeout check (e.g. 30 seconds to align)
            if time.time() - align_start_time > 60: # Extended timeout for total process
                rospy.logwarn("Arm alignment timed out!")
                return False

            res = None
            with self.lock:
                res = self.detect_result
                
            if res:
                cx = res['center_x']
                cy = res['center_y']
                color_id = res['data']
                
                # --- Stability Check ---
                if not position_en:
                    dx_pos = abs(cx - last_x)
                    dy_pos = abs(cy - last_y)
                    last_x = cx
                    last_y = cy
                    if dx_pos < 3 and dy_pos < 3:
                        stable_pos_count += 1
                        if stable_pos_count == 10:
                            stable_pos_count = 0
                            position_en = True
                    else:
                        stable_pos_count = 0
                else:
                    # --- Tracking Logic (Arm Only) ---
                    # X PID (Controls Servo 6)
                    self.x_pid.SetPoint = self.centreX
                    self.x_pid.update(cx)
                    dx = self.x_pid.output
                    
                    # Update Arm Base (Servo 6)
                    self.x_dis += int(dx)
                    self.x_dis = max(0, min(1000, self.x_dis))
                    
                    # Y PID (Controls Extension)
                    self.y_pid.SetPoint = self.centreY
                    self.y_pid.update(cy)
                    dy = self.y_pid.output
                    self.y_dis += dy
                    self.y_dis = max(0.12, min(0.28, self.y_dis))
                    
                    # Move Arm
                    target = self.ik.setPitchRanges((0, round(self.y_dis, 4), 0.03), -180, -180, 0)
                    if target:
                        servo_data = target[1]
                        bus_servo_control.set_servos(self.joints_pub, 20, (
                            (3, servo_data['servo3']), (4, servo_data['servo4']),
                            (5, servo_data['servo5']), (6, self.x_dis)
                        ))
                        
                        if len(target) > 2:
                            offset_y = Misc.map(target[2], -180, -150, -0.04, 0.03)
                    
                    # --- Stable Tracking Check ---
                    if abs(dx) < 2 and abs(dy) < 0.003:
                        stable_track_count += 1
                        if stable_track_count == 10:
                            stable_track_count = 0
                            
                            # --- Strict Color Mean Check (from intelligent_grasp_node) ---
                            color_buf.append(color_id)
                            if len(color_buf) == 5:
                                mean_num = np.mean(color_buf)
                                # Red:1, Green:2, Blue:3
                                # Check if mean matches target color ID roughly (float comparison)
                                # Or just if it is one of the valid IDs (1.0, 2.0, 3.0)
                                if mean_num == 1.0 or mean_num == 2.0 or mean_num == 3.0:
                                    # Double check if it matches OUR target color
                                    target_map = {'red': 1, 'green': 2, 'blue': 3}
                                    if target_map.get(color) == int(mean_num):
                                        rospy.loginfo(f"Grasp Confirmed: Mean Color ID {mean_num}")
                                        arm_move = True
                                    else:
                                        rospy.logwarn(f"Color Mismatch! Target: {color}, Detected Mean: {mean_num}")
                                        color_buf = [] # Reset buffer
                                else:
                                    color_buf = [] # Reset if noisy
                    else:
                        stable_track_count = 0
                        color_buf = [] # Reset buffer if movement unstable
            else:
                 # If lost, maybe wait?
                 rospy.sleep(0.05)
                        
            rospy.sleep(0.05)
        
        # Execute Grasp
        rospy.loginfo("Executing Grasp...")
        target = self.ik.setPitchRanges((0, round(self.y_dis + offset_y, 4), -0.08), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1500, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        bus_servo_control.set_servos(self.joints_pub, 800, ((1, 450),)) # Close
        rospy.sleep(0.8)
        
        # Lift Arm (Keep Rotation)
        bus_servo_control.set_servos(self.joints_pub, 2000, (
            (1, 450), (2, 500), (3, 80), (4, 825), (5, 625), (6, self.x_dis)
        ))
        rospy.sleep(1.5)
        # Don't reset arm to center yet, we need to turn right
        return True

    def find_and_place_tag(self, tag_id):
        rospy.loginfo(f"Searching for AprilTag {tag_id} (Arm Right)...")
        with self.lock:
            self.mode = 'apriltag'
            self.target_tag_id = tag_id
            self.detect_result = None
            
        # Turn Arm Base (Servo 6) to Right (e.g., 100-200 range)
        # Default 500. Right < 500.
        # Explicitly command Servo 6 to Right (200) before searching
        bus_servo_control.set_servos(self.joints_pub, 2000, ((6, 200),))
        rospy.sleep(1.5)
        
        # Wait for detection
        start_time = time.time()
        found = False
        while time.time() - start_time < 5:
            with self.lock:
                if self.detect_result and self.detect_result['data'] == tag_id:
                    found = True
                    break
            rospy.sleep(0.1)
            
        if not found:
             # Try searching by moving chassis slightly
             rospy.logwarn(f"Tag {tag_id} not found immediately, adjusting chassis...")
             
             # Search Sequence: Rotate Left -> Rotate Right -> Move Fwd -> Move Back
             search_moves = [
                 (0, 0, 0.3),   # Rotate Left
                 (0, 0, -0.6),  # Rotate Right (back past center)
                 (0, 0, 0.3),   # Center again
                 (40, 90, 0),   # Forward a bit
                 (40, 270, 0)   # Backward a bit
             ]
             
             for move in search_moves:
                 rospy.loginfo(f"Search move: {move}")
                 self.move_chassis(*move)
                 rospy.sleep(0.5)
                 self.stop_chassis()
                 rospy.sleep(1.0) # Wait for camera
                 
                 # Check again
                 with self.lock:
                    if self.detect_result and self.detect_result['data'] == tag_id:
                        found = True
                        break
             
             if not found:
                 rospy.logwarn(f"Tag {tag_id} STILL not found!")
                 return False
             else:
                 rospy.loginfo(f"Tag {tag_id} found after adjustment!")
        
        # Align (using Servo 6 for X, Extension for Y)
        rospy.loginfo(f"Aligning to Tag {tag_id}...")
        self.x_pid.clear()
        self.y_pid.clear()
        self.x_dis = 200 # Current approx position
        self.y_dis = 0.15
        
        # Move to Placing Height (Higher than Grasp)
        target = self.ik.setPitchRanges((0, 0.15, 0.05), -180, -180, 0)
        if target:
             servo_data = target[1]
             bus_servo_control.set_servos(self.joints_pub, 1500, (
                 (3, servo_data['servo3']), (4, servo_data['servo4']),
                 (5, servo_data['servo5'])
             ))
        rospy.sleep(1.0)

        stable_track_count = 0
        placed = False
        
        while not placed:
            res = None
            with self.lock:
                res = self.detect_result
                
            if res and res['data'] == tag_id:
                cx = res['center_x']
                cy = res['center_y']
                
                # X PID
                self.x_pid.SetPoint = self.centreX
                self.x_pid.update(cx)
                dx = self.x_pid.output
                self.x_dis += int(dx)
                self.x_dis = max(50, min(500, self.x_dis)) # Limit Right Side
                
                # Y PID
                self.y_pid.SetPoint = self.centreY
                self.y_pid.update(cy)
                dy = self.y_pid.output
                self.y_dis += dy
                self.y_dis = max(0.12, min(0.28, self.y_dis))
                
                # --- Chassis Movement Support (Tag) ---
                if self.y_dis >= 0.28:
                     # Tag too far (Right Side). Robot facing forward.
                     # Tag is at Right. Arm extended Right.
                     # Need to move Right (towards tag).
                     self.move_chassis(0, 270, 0)
                     rospy.sleep(0.1)
                     self.stop_chassis()
                elif self.y_dis <= 0.12:
                     # Tag too close. Move Left.
                     self.move_chassis(0, 90, 0)
                     rospy.sleep(0.1)
                     self.stop_chassis()
                
                target = self.ik.setPitchRanges((0, round(self.y_dis, 4), 0.05), -180, -180, 0)
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(self.joints_pub, 20, (
                        (3, servo_data['servo3']), (4, servo_data['servo4']),
                        (5, servo_data['servo5']), (6, self.x_dis)
                    ))
                
                if abs(dx) < 2 and abs(dy) < 0.003:
                    stable_track_count += 1
                    if stable_track_count == 20: # Make sure it's really stable
                        placed = True
                else:
                    stable_track_count = 0
            else:
                 rospy.sleep(0.05)
            
            # Timeout or manual break needed? 
            # For now, assume we find it.
            rospy.sleep(0.05)
            
        # Execute Place
        rospy.loginfo("Placing Block...")
        # Lower Arm
        target = self.ik.setPitchRanges((0, round(self.y_dis + 0.04, 4), -0.05), -180, -180, 0)
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(self.joints_pub, 1000, (
                (3, servo_data['servo3']), (4, servo_data['servo4']),
                (5, servo_data['servo5']), (6, self.x_dis)
            ))
        rospy.sleep(1.5)
        
        # Open Claw
        bus_servo_control.set_servos(self.joints_pub, 500, ((1, 150),))
        rospy.sleep(0.8)
        
        # Lift Arm
        bus_servo_control.set_servos(self.joints_pub, 1000, (
            (1, 150), (2, 500), (3, 80), (4, 825), (5, 625), (6, 500)
        ))
        rospy.sleep(1.5)
        self.reset_arm()
        return True

    def run_task(self):
        rospy.loginfo("Task Started")
        
        # 1. Initialize & Posture
        self.init_robot()
        
        # 2. Move to Work Area (Forward 3s)
        # Left: Blocks, Right: Tags
        rospy.loginfo("Moving to Work Area...")
        self.move_chassis(60, 90, 0) # Slow speed
        rospy.sleep(5.0) 
        self.stop_chassis()
        rospy.sleep(0.5)
        
        tasks = [('red', 1), ('blue', 2), ('green', 3)]
        
        for color, tag_id in tasks:
            while True:
                # 3. Search & Grasp (Left Side)
                if self.search_color(color):
                    if self.align_and_grasp(color):
                        
                        # 4. Turn Arm to Right & Place (Right Side)
                        if self.find_and_place_tag(tag_id):
                            # Success! Break to next color
                            break
                        else:
                            rospy.logwarn(f"Failed to place {color}. Retrying sequence...")
                    else:
                        rospy.logwarn(f"Failed to grasp {color}. Retrying...")
                else:
                    rospy.logwarn(f"Could not find {color} block. Retrying...")
                
                # Small delay before retry to prevent crazy loops
                rospy.sleep(1.0)
                
        # 5. Return to Start
        rospy.loginfo("Returning to Start...")
        
        # Forward to Crossroad
        self.move_chassis(40, 90, 0)
        rospy.sleep(2.0)
        self.stop_chassis()
        rospy.sleep(0.5)
        
        # Turn Left (Face Start/Parking)
        # Wait, if I came from Start, turned Left to Blocks.
        # So Start is "South", Blocks is "West", Tags is "East".
        # Now I am at Center facing West (towards Blocks).
        # I need to go South. So Turn Left.
        self.move_chassis(0, 0, 0.5) # Turn Left 90
        rospy.sleep(1.2)
        self.stop_chassis()
        rospy.sleep(0.5)
        
        # Forward to Parking
        self.move_chassis(40, 90, 0)
        rospy.sleep(3.0)
        self.stop_chassis()
        
        rospy.loginfo("Task Completed")
        
    def start(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        task = IntegratedTransportTask()
        task.start()
    except rospy.ROSInterruptException:
        pass

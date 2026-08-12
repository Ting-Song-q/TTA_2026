#!/usr/bin/python3
# coding=UTF-8

from fabric2 import Connection
import rospy
from geometry_msgs.msg import Twist, Quaternion, PoseWithCovarianceStamped
from cv_bridge import CvBridge 
from tf.transformations import quaternion_from_euler,euler_from_quaternion
import tf
import actionlib
from actionlib_msgs.msg import *
from sensor_msgs.msg import LaserScan,Image
from move_base_msgs.msg import MoveBaseAction,MoveBaseGoal
import math
import numpy as np
# from Vision import Capture
import os
import cv2
import time
import copy
import threading
from threading import Thread
from ultralytics import YOLO
from pathlib import Path
import glob
from queue import Queue
class Model:
    def __init__(self, model_path, model_input_path, model_output_path,pic_recognize_count,num_workers=4):
        # self.model = YOLO(model_path)
        # this will bring some threading issues
        self.pic_recognize_lock = threading.Lock()
        self.pic_recognize_count = pic_recognize_count

        self.model_path = model_path
        self.model_input_path = model_input_path
        self.model_output_path = model_output_path
        self.model_input_buffer = Queue()
        self.model_output_buffer = Queue()

        self.condition = threading.Condition()
        self.threads_pool = []
        for i in range(0,num_workers):
            worker = Thread(target=self.work)
            worker.daemon = True
            worker.start()
            self.threads_pool.append(worker)
            

        self.PLANE_1 = False
        self.PLANE_2 = False
        self.CAR_RECOGNIZE = False
        self.CAR_FINISH = False

    def plane_run(self):
        while True:
            time.sleep(3)
            check = self.plane_task()
            if check:
                with self.condition:
                    self.condition.notify_all()
        
            if self.PLANE_1 and self.PLANE_2:
                return True
    
    def car_run(self):
        while True:
            time.sleep(5)
            if self.CAR_RECOGNIZE:
                self.CAR_RECOGNIZE = False
                with self.condition:
                    self.condition.notify_all()

            if self.CAR_FINISH:
                return True

    def work(self):
        while True:
            with self.condition:
                self.condition.wait()
                while not self.model_input_buffer.empty():
                    my_list = self.model_input_buffer.get()
                    img_name = my_list[0]
                    #print("recognize:!!!",img_name,my_list[1],type(my_list[1]))
                    recognize_result = self.recognize(my_list[1])
                    txt_line = [img_name, recognize_result]
                    with self.pic_recognize_lock:
                        self.pic_recognize_count -= 1
                    self.model_output_buffer.put(txt_line)
                    #print("APPEND A TXT LINE:",txt_line)
        


    def recognize(self,model_input):
        """_summary_

        Args:
            input (_type_): 可以是cv2.imread()返回的图片Mat(numpy ndarray数组),也可以是图片路径,还可以是一些别的东西.

        Returns:
            _type_: 推理结果list,例如['1 good 1 bad']
        """
        model = YOLO(self.model_path)
        results = model(model_input)


        if len(results) == 1 :
            print("yesyesyes finish recognize recognize!")

        else:
            print("it seems that you give me some wrong inputs.")


        for r in results:
            res = r.verbose()
        return res

    # 仅供飞机使用的方法
    def plane_task(self):
        file_names = os.listdir(self.model_input_path)
        if '1440x1088_1.txt' in file_names and self.PLANE_1 == False:
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1' + '*')))
            for i in image_paths:
                image = cv2.imread(i)
                self.append(image,i)
            self.PLANE_1 = True
            #print(self.model_output_buffer.qsize())
            return True

        if '1440x1088_2.txt' in file_names and self.PLANE_2 == False:
            image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_2' + '*')))
            for i in image_paths:
                image = cv2.imread(i)
                self.append(image,i)
            self.PLANE_2 = True
            #print(self.model_output_buffer.qsize())
            return True 
            
        # if '1440x1088_1-top.txt' in file_names and self.top_1 == False:
        #     image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1-top' + '*')))
        #     for i in image_paths:
        #         image = cv2.imread(i)
        #         self.append(image,i)
        #     self.top_1 = True
        #     print(self.model_output_buffer.qsize())
        #     return True

        # if '1440x1088_1-middle.txt' in file_names and self.middle_1 == False:
        #     image_paths = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1-middle' + '*')))
        #     for i in image_paths:
        #         image = cv2.imread(i)
        #         self.append(image,i)
        #     self.middle_1 = True
        #     print(self.model_output_buffer.qsize())
        #     return True
        return False


            
    def append(self,image,image_name):
        self.model_input_buffer.put([image,image_name])
        return True
    

    def save(self):
        print("enter save def")
        while self.model_output_buffer.qsize() !=0:
            txt_line = self.model_output_buffer.get()
            print("this is a saving txt line?",txt_line)
            with open(self.model_output_path,'a') as f:
                f.write(','.join(txt_line) + '\n')
                print("Save Result:",txt_line)     

        return True




class Nav:
    def __init__(self):
        rospy.init_node('nav', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_subscriber = rospy.Subscriber('/scan', LaserScan, self.laser_callback,queue_size=1)
        self.image_raw = rospy.Subscriber("ep_cam/image_raw", Image, self.camera_cb, queue_size=1)
        self.cv_bridge = CvBridge()
        self.laser_data = None
        self.image = None
        self.take_picture_need = False
        self.image_lock = threading.Lock()

        self.plane_model = Model("/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-1.pt",
        model_input_path='/home/tta/Project_ws/src/rmep_nav/scripts/image_from_flight/',
        model_output_path='/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result.txt',
        pic_recognize_count=16)

        self.car_model = Model("/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-2.pt",
        model_input_path = '/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/',
        model_output_path='/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result.txt',
        pic_recognize_count=8)
        
        self.car_model_manager = Thread(target=self.car_model.car_run)
        self.car_model_manager.start()
        # self.plane_model_manager = Thread(target=self.plane_model.plane_run)
        # self.plane_model_manager.start()

        self.rate = rospy.Rate(150)
        self.tf_listener = tf.TransformListener() 
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while(not self.move_base.wait_for_server(rospy.Duration(1.0))):
            continue
        self.nav_for_projecte()
        
    def nav_for_projecte(self):
        self.move_to_initial_position(0.75, 0.65) 
        self.initial_pose_for_amcl()

        # self.is_go('/mnt/start.txt')
        # print("isgo!!!!!!!!!!!!!!")
        
        self.turn_ang(0.5,math.pi/2)
        rospy.sleep(1)

        time = rospy.Time.now()
        self.go_picture_point(2.0,0.45,0,'1')

        time = rospy.Time.now()
        self.go_picture_point(3.5,0.45,0,'2')

        time = rospy.Time.now()
        self.go_picture_point(4.75,0.45,0,'3')

        time = rospy.Time.now()
        self.go_picture_point(5.75,0.45,0,'4')

        #
        self.car_model.CAR_RECOGNIZE = True
        #


        a = self.make_move_base_goal(7.25,0.45,math.pi/2)
        self.move(a,math.pi/2,False)
        rospy.sleep(1)

        b = self.make_move_base_goal(7.25,2.7,math.pi/2)
        self.move(b,math.pi/2,True)
        rospy.sleep(1)

        # self.has_arrive('/mnt/arrival.txt')
        # self.is_go('/mnt/start_again.txt')

        self.turn_ang(-0.5,math.pi/2)

        c = self.make_move_base_goal(0,2.75,0)
        self.move(c,0,False)
        rospy.sleep(1)

        time = rospy.Time.now()
        self.go_picture_point(2,2.75,0,'5')

        time = rospy.Time.now()
        self.go_picture_point(3.5,2.75,0,'6')
       
        time = rospy.Time.now()
        self.go_picture_point(4.75,2.75,0,'7')

        time = rospy.Time.now()
        self.go_picture_point(5.75,2.75,0,'8')

        #
        self.car_model.CAR_RECOGNIZE = True
        #

        self.turn_ang(0.5,math.pi)
        rospy.sleep(1)

        d = self.make_move_base_goal(0,2.75,-math.pi/2)
        self.move(d,-math.pi/2,False)
        rospy.sleep(1)

        destinition = self.make_move_base_goal(0.25,0.25,math.pi/2)
        self.move(destinition,math.pi/2,True)
        rospy.sleep(1)

        print("start to finish 2222222222222222222222222222222222222222222222222222222222222222")
        self.finish()


    def go_picture_point(self,x,y,yaw,image_name):
        point = self.make_move_base_goal(x,y,yaw)
        self.move(point,yaw,True)
        self.take_picture(image_name)
    

    def camera_cb(self, msg):
        with self.image_lock:
            if self.take_picture_need:
                self.image = self.cv_bridge.imgmsg_to_cv2(msg,"bgr8")
            else:
                return
        
    def take_picture(self, image_name):
        self.take_picture_need = True
        while self.image is None:
            continue
        rospy.sleep(1.5)

        image = copy.deepcopy(self.image)
        # image = self.image
        # / 有无必要使用deepcopy？
        dist_coeffs = np.array([-0.0444954569665007, -0.00201770323876866, 0, 0, -0.0159863040899372])
        intrinsic = np.array([[624.302683276108, 0, 632.376721286788],
                        [0, 623.915644264347, 371.420934824104],
                        [0, 0, 1]])
        corrected_image = cv2.undistort(image, intrinsic, dist_coeffs)     
        # 可以注释掉的deepcopy，理论上是为了防止脏写，但是OS或cv2包可能已经进行了维护   
        #
        cv2.imwrite(f'/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/{image_name}.jpg',corrected_image)
        # 

        self.car_model.append(image_name,corrected_image)
        self.take_picture_need = False

        time.sleep(0.5)
        self.image = None
        
    def has_arrive(self,path):
        hostname = '192.168.31.110'
        username = 'root'
        password = '123456'
        filepath = path

        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})

        conn.run(f"touch {filepath}")

    def is_go(self,path):
        hostname = '192.168.31.110'
        username = 'root'
        password = '123456'
        filepath = path

        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})

        while True:
            try:
                time.sleep(0.5)
                result = conn.run('test -e {}'.format(filepath), hide=True)
                if result.ok: 
                    conn.run(f'rm {filepath}')
                    conn.close()
                    break
            except Exception as e:
                print("continue to check")
                continue

    def initial_pose_for_amcl(self):
        initialpose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        initial_pose = PoseWithCovarianceStamped()

        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.header.frame_id = "map"
        
        initial_pose.pose.pose.position.x = 0.0
        initial_pose.pose.pose.position.y = 0.0
        initial_pose.pose.pose.position.z = 0.0
        
        quaternion = quaternion_from_euler(0, 0, math.pi/2) 
        initial_pose.pose.pose.orientation.x = quaternion[0]
        initial_pose.pose.pose.orientation.y = quaternion[1]
        initial_pose.pose.pose.orientation.z = quaternion[2]
        initial_pose.pose.pose.orientation.w = quaternion[3]

        initial_pose.pose.covariance = [0.25, 0, 0, 0, 0, 0,
                                        0, 0.25, 0, 0, 0, 0,
                                        0, 0, 0.25, 0, 0, 0,
                                        0, 0, 0, 0.068, 0, 0,
                                        0, 0, 0, 0, 0.068, 0,
                                        0, 0, 0, 0, 0, 0.068]
        start_time = rospy.Time.now() 
        while (rospy.Time.now() - start_time).to_sec() < 0.5:
            initialpose_pub.publish(initial_pose)
            self.rate.sleep()
        rospy.sleep(1)
        print("initialpose success!")


    def laser_callback(self, laser_data):
        self.laser_data = laser_data

    def move_to_initial_position(self, goal_left, goal_back):
        self.align_wall()
        retries_y = 0
        retries_x = 0
        while not rospy.is_shutdown():
            if self.laser_data:
                laser_data = self.laser_data
                left_distance = self.get_distance(laser_data,-90)
                back_distance = self.get_distance(laser_data,0)
                if left_distance == float('inf') or back_distance == float('inf'):
                    continue
                if abs(left_distance - goal_left) > 0.02 and retries_y < 3:
                    error_y = abs(left_distance - goal_left)
                    speed = - 0.1 * (left_distance - goal_left) / abs(left_distance - goal_left)  
                    self.go_linear_y(speed,error_y)
                    retries_y += 1
                    rospy.sleep(1)
                    continue
                if abs(goal_back - back_distance) > 0.02 and retries_x < 3:
                    error_x = abs(goal_back - back_distance)
                    speed = 0.1 * (goal_back - back_distance) / abs(goal_back - back_distance)
                    self.go_linear_x(speed,error_x)
                    retries_x += 1
                    rospy.sleep(1)
                    continue
                # self.stop()
                break
            self.rate.sleep()
        self.align_wall()
        laser_data = self.laser_data
        left_distance = self.get_distance(laser_data,-90)
        back_distance = self.get_distance(laser_data,0)
        print(left_distance,back_distance)

    def get_distance(self, laser_data, angle):
        angle_in_rad = math.radians(angle)
        dis = laser_data.ranges[int((angle_in_rad - laser_data.angle_min) / laser_data.angle_increment)]
        return dis

    def align_wall(self):
        count = 0
        while not rospy.is_shutdown():
            if self.laser_data:
                try:
                    laser_data = self.laser_data
                    THETA = math.pi / 180 * 30
                    b = self.get_distance(laser_data, -90)
                    a = self.get_distance(laser_data, -120)
                    alpha = -math.atan((a * math.cos(THETA) - b) / (a * math.sin(THETA)))
                    if b == float('inf') or a == float('inf') or alpha == np.nan:
                        continue
                    ab = b * math.cos(alpha)
                    print(alpha, ab)
                    if abs(alpha) > 0.01 and count < 3:
                        self.turn_ang(0.6,alpha)
                        count += 1
                        rospy.sleep(1)
                    else:
                        self.stop()
                        rospy.sleep(1)
                        print("??????????????")
                        break
                except Exception as e:
                    continue
            # self.rate.sleep()

    def stop(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        vel_msg.linear.y = 0
        self.velocity_publisher.publish(vel_msg)

    def turn_ang(self, ang_speed, goal_rotation):
        twist_ang = Twist()
        twist_ang.angular.z = -ang_speed if goal_rotation > 0 else ang_speed
        twist_ang.linear.x = 0
        twist_ang.linear.y = 0
        target_rotation_time = math.fabs(goal_rotation / ang_speed)
        turn_start_time = rospy.Time.now()
        while (rospy.Time.now() - turn_start_time).to_sec() < target_rotation_time:
            self.velocity_publisher.publish(twist_ang)
            self.rate.sleep()
        twist_ang.angular.z = 0
        self.velocity_publisher.publish(twist_ang)
        
    def go_linear_x(self,linear_speed,goal_distance):
        target_linear_time = math.fabs(goal_distance /linear_speed)
        twist_linear = Twist()
        twist_linear.linear.x = linear_speed
        twist_linear.linear.y = 0
        twist_linear.angular.z = 0
        go_start_time = rospy.Time.now() 
        while (rospy.Time.now() - go_start_time).to_sec() < target_linear_time:
            self.velocity_publisher.publish(twist_linear)
            self.rate.sleep()
        twist_linear.linear.x = 0
        self.velocity_publisher.publish(twist_linear)

    def go_linear_y(self,linear_speed,goal_distance):
        target_linear_time = math.fabs(goal_distance /linear_speed)
        twist_linear = Twist()
        twist_linear.linear.x = 0
        twist_linear.linear.y = linear_speed
        twist_linear.angular.z = 0
        go_start_time = rospy.Time.now() 
        while (rospy.Time.now() - go_start_time).to_sec() < target_linear_time:
            self.velocity_publisher.publish(twist_linear)
            self.rate.sleep()
        twist_linear.linear.y = 0
        self.velocity_publisher.publish(twist_linear)


    def move(self, goal, target_yaw, is_xy_fix, timeout=25, retries=3):
        attempt = 0
        while attempt < retries:
            attempt += 1
            self.move_base.send_goal(goal)
            success = self.move_base.wait_for_result(rospy.Duration(timeout))        
            if success:
                state = self.move_base.get_state()
                if state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Goal reached successfully on attempt %d", attempt)
                    rospy.sleep(2)
                    self.adjust_pose(target_yaw)
                    rospy.sleep(1.5)
                    if is_xy_fix == False:
                        return
                    rate = rospy.Rate(20)
                    while not rospy.is_shutdown():
                        try:
                            (current_postion,rotation) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
                            rate.sleep()
                            break
                        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                            continue
                    current_position_x = current_postion[0]
                    current_position_y = current_postion[1]
                    error_x = current_position_x - goal.target_pose.pose.position.x
                    error_y = current_position_y - goal.target_pose.pose.position.y
                    if abs(error_x) > 0.025:
                        self.go_linear_x(-0.1*abs(error_x)/(error_x),abs(error_x))
                        rospy.sleep(0.5)
                    if abs(error_y) > 0.025:
                        if current_position_y > goal.target_pose.pose.position.y:
                            self.turn_ang(0.6,math.pi/2)
                            rospy.sleep(0.5)
                            self.go_linear_x(0.1,current_position_y - goal.target_pose.pose.position.y)
                            rospy.sleep(0.5)
                            self.turn_ang(-0.6,math.pi/2)
                            rospy.sleep(1)
                        else:
                            self.turn_ang(-0.6,math.pi/2)
                            rospy.sleep(0.5)
                            self.go_linear_x(0.1,-(current_position_y - goal.target_pose.pose.position.y))
                            rospy.sleep(0.5)
                            self.turn_ang(0.6,math.pi/2)
                            rospy.sleep(1)
                    self.adjust_pose(target_yaw)
                    rospy.sleep(1.5)
                    # while not rospy.is_shutdown():
                    #     try:
                    #         (current_postion2,rotation2) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
                    #         rate.sleep()
                    #         break
                    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    #         continue
                    # current_position_x2 = current_postion2[0]
                    # current_position_y2 = current_postion2[1]
                    # print("is_xy_fix_after:",current_position_x2,current_position_y2)
                    return True
                else:
                    rospy.logwarn("Attempt %d: Goal failed with state: %d", attempt, state)
            else:
                rospy.logwarn("Attempt %d: Goal timed out after %d seconds", attempt, timeout)
            
            rospy.loginfo("Retrying... (attempt %d/%d)", attempt, retries)
        
        rospy.logerr("All attempts failed. Unable to reach the goal.")
        return False
    
    def make_move_base_goal(self, x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        q = quaternion_from_euler(0,0,yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        return goal     

    def adjust_pose(self,target_yaw):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                (current_postion,current_rotation) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
                rate.sleep()
                break 
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        current_position_x = current_postion[0]
        current_position_y = current_postion[1]

        _, _, current_yaw = euler_from_quaternion(current_rotation)
        error = current_yaw - target_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        if abs(error) > 0.025:
            self.turn_ang(0.6*abs(error)/error,abs(error))


    
    def finish(self):
        
        while True:
            time.sleep(1)
            if self.car_model.pic_recognize_count==0:
                self.car_model.CAR_FINISH = True
                break
        
        # while True:
        #     time.sleep(1)
        #     if self.plane_model.PIC_RECOGNIZE_COUNT==0:
        #         break
        
        #self.plane_model_manager.join()
        self.car_model_manager.join()

        print("FINISH CAR RECOGNITION!")

        # print('FINISH PLANE RECOGNITION!')
        print("START TO SAVE A TXT")
        self.car_model.save()
        #self.plane_model.save()
        print("SAVE AS RESULT.TXT")
        return True

if __name__ == '__main__':
    nav = Nav()
    rospy.spin()



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
import csv
import torch
import re
import pysftp

def num_to_fruit(num):
    fruit=None
    if(num==0):
        fruit="apple"
    elif(num==1):
        fruit="banana"
    elif(num==2):
        fruit="orange"
    elif(num==3):
        fruit="watermelon"
    elif(num==4):
        fruit="other"
    else:
        print("未找到对应水果，输入水果编号有误")
    return fruit

def process_outputs(detections, conf_threshold=0.75):
    """
    处理 YOLOv5 的检测结果，筛选出符合置信度阈值的检测框。
    
    参数:
        detections: YOLOv5 的 Detections 对象
        conf_threshold: 置信度阈值，默认为 0.5
    
    返回:
        list: 筛选后的检测框，每个框为字典，包含归一化坐标、置信度和类别
    """
    xywh_list = []
    conf_list = []
    cls_list = []
    
    
    # 获取图像的宽度和高度
    img_height = 1280 
    img_width = 720

    # 遍历检测结果
    for det in detections.xyxy[0]:  # 遍历第一个 batch 的检测结果
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()  # 转为 numpy 格式
        if conf >= conf_threshold:  # 筛选置信度符合阈值的框
            xywh_list.append([
                    ((x1 + x2) / 2) / img_width,  # 中心 x 坐标归一化
                    ((y1 + y2) / 2) / img_height,  # 中心 y 坐标归一化
                    abs(x2 - x1) / img_width,   # 宽度归一化
                    abs(y2 - y1) / img_height   # 高度归一化
            ])
            conf_list.append(conf)  # 置信度
            cls_list.append(int(cls))  # 类别
            print(conf_list)
    return xywh_list, cls_list, conf_list
    


def filter_boxes(xywh_list, cls_list, conf_list, span_factor=1.25, layer=1, min_factor=0.2):
    """
    过滤和统计平面上的检测框
    
    参数:
        boxes: 包含检测框信息的对象，应有xywhn和cls属性
        span_factor: 滑动窗口的扩展因子，默认为1.25
        layer: 检测层级(1或2)，默认为1
        min_factor: 最小面积因子，用于过滤小框，默认为0.2
    
    返回:
        tuple: (是否有效, 置信度最高的种类)
    """

    BEST_CLS  = None

    # 1. 如果没有识别到物体，返回False,并且随机选择一个种类
    if len(cls_list) == 0:
        print("I never saw any fruit")
        return False,4
    elif len(cls_list) == 1:
        print("I only saw one fruit")
        BEST_CLS = cls_list[0]
        if -0.01 < BEST_CLS < 0.01:
            BEST_CLS = 0  # 苹果
        elif -0.01 < BEST_CLS - 1  < 0.01:
            BEST_CLS = 1  # 香蕉
        elif -0.01 < BEST_CLS - 2  < 0.01:
            BEST_CLS = 2  # 橘子
        elif -0.01 < BEST_CLS - 3  < 0.01:
            BEST_CLS = 3  # 西瓜
        else:
            print("I hope that I would never see this output")
            
        print("I believe it is a/an {}".format(num_to_fruit(BEST_CLS)))
        return True, BEST_CLS
    # 2. 如果识别到物体
    else:
        print("I saw {} fruit".format(len(cls_list)))
        print("!!!!!!!")
        print(xywh_list)
        print(cls_list)

        min_distance = 100
        for obj_box, obj_cls, obj_conf in zip(xywh_list, cls_list, conf_list):
            obj_width, obj_height = obj_box[2], obj_box[3]  #获取识别框的宽高
            obj_central_x, obj_central_y = obj_box[0], obj_box[1]

            if layer == 1:  # 与顶部右边(1,0)的距离
                my_distance = (obj_central_x-1)**2 + (obj_central_y)**2
            elif layer == 2:    # 与底部右边(1,1)的距离
                my_distance = (obj_central_x-1)**2 + (1-obj_central_y-1)**2
            else:           #与中间最近的距离
                my_distance = (obj_central_x-0.5)**2 + (obj_central_y-0.5)**2

            if my_distance < min_distance:
                min_distance = my_distance
                BEST_CLS = obj_cls


        if -0.01 < BEST_CLS < 0.01:
            BEST_CLS = 0  # 苹果
        elif -0.01 < BEST_CLS - 1  < 0.01:
            BEST_CLS = 1  # 香蕉
        elif -0.01 < BEST_CLS - 2  < 0.01:
            BEST_CLS = 2  # 橘子
        elif -0.01 < BEST_CLS - 3  < 0.01:
            BEST_CLS = 3  # 西瓜
        else:
            print("I hope that I would never see this output")
            

        print("I believe it is a/an {}".format(BEST_CLS))
        return True, BEST_CLS

  


    
    
import threading
import time
from queue import Queue
from threading import Thread
import os
import glob
import re
# 注意：以下导入在实际代码中需要取消注释
# from ultralytics import YOLO
# from utils import filter_boxes_plane, filter_boxes_car  # 假设有自定义工具函数

class Model:
    def __init__(self, device, model_path, model_input_path, pic_recognize_count, num_workers=4):
        """
        初始化模型推理管理器
        
        参数:
            device: 设备类型 ('plane' 或 'car')
            model_path: YOLO模型文件路径
            model_input_path: 输入图片的根目录
            pic_recognize_count: 待识别的图片总数（用于计数）
            num_workers: 工作线程数 (默认4)
        """
        # 模型相关路径
        self.model_path = model_path
        self.model_input_path = model_input_path
        
        # 任务队列：输入队列存储待识别图片，输出队列存储识别结果
        self.model_input_buffer = Queue()
        self.model_output_buffer = Queue()
        
        self.csv_time = rospy.Time.now()
        self.csv_writer = CsvWriter(self.model_output_buffer,
                            f'/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result_car.csv')
        self.device = device
        self.pic_recognize_lock = threading.Lock()  # 图片计数线程锁
        self.pic_recognize_count = pic_recognize_count  # 剩余待识别图片计数器




        # 线程同步机制
        self.condition = threading.Condition()  # 条件变量控制任务调度
        self.threads_pool = []  # 工作线程池
        
        # 初始化工作线程 (守护线程)
        for i in range(0, num_workers):  # 创建4个工作线程
            worker = Thread(target=self.work)  # 设置线程目标为work方法
            worker.daemon = True  # 主线程退出时自动终止
            worker.start()  # 启动线程
            self.threads_pool.append(worker)

        self.CAR_RECOGNIZE = False
        self.CAR_FINISH = False
            
        

    def car_run(self):
        while True:
            time.sleep(5)
            if self.CAR_RECOGNIZE:
                self.CAR_RECOGNIZE = False
                with self.condition:
                    self.condition.notify_all()

            if self.CAR_FINISH:
                print("finish car task")
                return True
         

    def work(self):
        """工作线程核心函数：从队列取任务并执行识别"""
        while True:
            with self.condition:
                self.condition.wait()  # 等待任务通知
                # 处理输入队列中的所有任务
                while not self.model_input_buffer.empty():
                    task = self.model_input_buffer.get()
                    img_name = task[0]
                    img_data = task[1]  # 可能是路径或图像数据
                    
                    # 执行识别并获取结果
                    ret, cls = self.recognize(img_data)
                    
                    # 更新剩余图片计数 (线程安全)
                    with self.pic_recognize_lock:
                        self.pic_recognize_count -= 1
                    
                    # 将结果存入输出队列 (文件名, 类别)
                    result_line = (img_name, cls)
                    self.model_output_buffer.put(result_line)
                    self.csv_writer.yolo_write()

                    if self.pic_recognize_count == 0:
                        self.download_flight_result('/home/forlinx/catkin_ws/src/tta_m3e_rtsp/model_result/flight_result.csv',
                                                    '/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result_flight.csv',
                                                    '/home/forlinx/catkin_ws/src/tta_m3e_rtsp/my_msg/result.txt')

                        flight_csv = "/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result_flight.csv"
                        car_csv = "/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result_car.csv"
                        output_csv = "/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result.csv" 

                        merge_csv_files(flight_csv, car_csv, output_csv)
                        print(f"合并完成！结果已保存至: {output_csv}")



    def recognize(self, model_input):
        """
        执行单张图片推理
        
        参数:
            model_input: 输入数据 (图片路径/图像数组)
        返回:
            tuple: (合格数量, 缺陷数量)
        """
        # 动态加载模型 (注意：频繁加载可能影响性能)
        model = torch.hub.load('/home/tta/Project_ws/src/rmep_nav/src/yolov5','custom',path=self.model_path, source='local',device='cpu')
        
        # 执行推理 (置信度阈值0.75)
        results = model(model_input)
        res_xywh,res_cls,res_conf = process_outputs(results)  # 获取检测框结果
        
        # 设备类型分支处理
        try:
                # 从文件名解析层级信息 (示例: "XXX_1-3.jpg")
            ret, cls = filter_boxes(res_xywh,res_cls,res_conf,layer=0)

            return ret,cls  # 选择与中部最近的

                
        except Exception as e:
            print(f"Error in recognize: {str(e)}")
            return 0, 0  # 返回默认值


    def append(self, image_name, image):
        """
        添加任务到输入队列
        
        参数:
            image_name: 图片标识 (通常为路径)
            image: 图片数据 (此处直接使用路径)
        返回:
            bool: 总是True
        """
        self.model_input_buffer.put([image_name, image])
        return True
    

    def download_flight_result(self,remote_path, local_path, msg_path):
        # 配置信息
        hostname = '192.168.31.66'
        username = 'forlinx'
        password = 'forlinx'
        conn = None

        while True:
            try:
                # 建立连接
                conn = Connection(host=hostname, user=username, 
                                connect_kwargs={"password": password})
                
                while True:
                    # 检查标记文件是否存在
                    result = conn.run(f'test -e {msg_path}', hide=True)
                    if result.ok:
                        print("标记文件存在，开始读取CSV内容...")
                        
                        # 方法1：直接使用cat命令读取CSV内容
                        while True:
                            cat_result = csv_result = conn.run(f'cat {remote_path}', hide=True)
                            if cat_result.ok:
                                csv_content = csv_result.stdout
                                print("读取成功")
                                break
                            else:
                                print("读取不成功，等待2秒后重试...")
                                time.sleep(2)
                        
                        # 将内容写入本地文件
                        with open(local_path, 'w') as f:
                            f.write(csv_content)
                        
                        # 删除标记文件
                        conn.run(f'rm {msg_path}')
                        conn.close()
                        print("CSV内容已成功读取并保存到本地")
                        return    
                    else:
                        print("标记文件不存在，等待2秒后重试...")
                        time.sleep(2)
                        
            except Exception as e:
                print(f"连接出错, 尝试重新连接...")
                if conn:
                    conn.close()
                time.sleep(2)
        

    


class CsvWriter:
    def __init__(self,model_buffer,output_file_path):
        self.model_buffer = model_buffer
        self.output_file_path = output_file_path
        with open(output_file_path,mode='w',newline='') as f:
            init_writer = csv.writer(f)
            init_writer.writerow(('I区货架号','水果种类','II区货架号','水果种类')) #设置列名
            for i in range(1, 13):  # 货架1-12   
                row = (
                    f'="1-{i}"',
                    None,
                    f'="2-{i}"',
                    None
                )
                init_writer.writerow(row)
                print("row{}:".format(i), row)
                print("init over")
        f.close()
    
    
    def yolo_write(self):  
        csv_file = self.output_file_path
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)  # 所有行转为列表
        # 当缓冲区中有数据时循环处理
        while self.model_buffer.qsize() > 0:
            # 从缓冲区获取一个元组数据
            model_tuple = self.model_buffer.get()
            shelf_id = model_tuple[0]
            match = re.search(r'(\d+)-(\d+)', shelf_id)
            if match:
                # 将匹配到的部分用下划线连接
                shelf_id = f"{match.group(1)}_{match.group(2)}"
                print(shelf_id)
                row = match.group(2)
                if match.group(1) == '1':
                    rows[int(row)][1] = num_to_fruit(model_tuple[1])
                elif match.group(1) == '2':
                    rows[int(row)][3] = num_to_fruit(model_tuple[1])
                else:
                    print("No match shelf")
            else:
                # 未找到匹配格式时打印提示
                print("No match found")

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        print("yolo writer over")
        f.close()
                


    def static_write(self, fruit_1, shelf_1, fruit_2, shelf_2):
        csv_file = self.output_file_path
        with open(csv_file, mode='a', newline='') as f:
            static_writer = csv.writer(f)
            for i in range(1, 13):  # 货架1-12
                # 直接比较货架号
                shelf1_fruit = num_to_fruit(fruit_1) if i == shelf_1 else None
                shelf2_fruit = num_to_fruit(fruit_2) if i == shelf_2 else None
                
                # 构造单行数据（元组）
                row = (
                    f'="1-{i}"',
                    None,
                    f'="2-{i}"',
                    None
                )
                static_writer.writerow(row)
                print("row{}:".format(i), row)
        print("static writer over")
        f.close()



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


        self.car_model = Model('car',
            "/home/tta/Project_ws/src/rmep_nav/src/yolov5/best_0814.pt",
        model_input_path='/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/',
        pic_recognize_count=8)

        # self.plane_model = Model('plane',
        #     "/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-2.pt",
        # model_input_path='/home/tta/Project_ws/src/rmep_nav/scripts/image_from_flight/',
        # pic_recognize_count=16)

        # self.car_model = Model('car',
        #     "/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-1.pt",
        # model_input_path = '/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/',
        # pic_recognize_count=8)
        
        self.car_model_manager = Thread(target=self.car_model.car_run)
        self.car_model_manager.start()
        # self.plane_model_manager = Thread(target=self.plane_model.plane_run)
        # self.plane_model_manager.start()
        # self.csv_time = rospy.Time.now()
        # self.csv_writer = CsvWriter(self.car_model.model_output_buffer,
        #                     self.plane_model.model_output_buffer,
        #                     f'/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result{self.csv_time}.csv')
    
        self.rate = rospy.Rate(100)
        self.tf_listener = tf.TransformListener() 
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while(not self.move_base.wait_for_server(rospy.Duration(1.0))):
            continue
        self.nav_for_projecte()
        
    def nav_for_projecte(self):
        #go_picture是最高层次的封装，调用make_move_base_goal生成导航目标
        #调用move将导航目标传入move_base进行移动

        print("nav start")

        bioas = 0
        
        # def move_to_initial_position(self, goal_left, goal_back)
        #self.move_to_initial_position(2.3, 0.9)
        #self.move_to_initial_position(2.3, 0.9)

        #x,y,z,yaw
        self.initial_pose_for_amcl(0,0,0,0)


        #车辆到达一区
        a1=self.make_move_base_goal(2.5+0.20,0.9-0.10,0)
        self.move(a1,0,True)
        rospy.sleep(2)

        #通信，飞机起飞
        # self.please_takeoff('/home/forlinx/catkin_ws/src/tta_m3e_rtsp/my_msg/takeoff.txt')
        print("please take off!!!!!!!!!!!")
        rospy.sleep(2)

        #一货架拍照
        #self.go_picture_point(1.75+bioas,0.9,0,'1-9')
        self.go_picture_point(3.25+bioas,0.9-0.10,0,'1-10')
        self.go_picture_point(4.5+bioas,0.9-0.10,0,'1-11')
        self.go_picture_point(5.5+bioas,0.9-0.10,0,'1-12')
        self.car_model.CAR_RECOGNIZE=True

        #二货架拍照
        self.go_picture_point(5.5+bioas,-0.9,math.pi,'2-12')
        self.go_picture_point(4.5+bioas,-0.9,math.pi,'2-11')
        self.go_picture_point(3.25+bioas,-0.9,math.pi,'2-10')
        self.go_picture_point(1.75+bioas,-0.9,math.pi,'2-9')
        self.go_picture_point(1.75+bioas,0.9,0,'1-9')
        self.car_model.CAR_RECOGNIZE=True

        #车辆到达二区
        a2=self.make_move_base_goal(5,-0.9,0)
        self.move(a2,0,True)
        rospy.sleep(2)
        # self.please_landing('/home/forlinx/catkin_ws/src/tta_m3e_rtsp/my_msg/landing.txt')

        #通信，飞机降落，车辆驶往充电区
        # self.is_go_charge('/home/forlinx/catkin_ws/src/tta_m3e_rtsp/my_msg/go_charge.txt', 300)
        # rospy.sleep(2)
        b=self.make_move_base_goal(7,0,0)
        self.move(b,0,False)
        rospy.sleep(2)

        #返航
        c=self.make_move_base_goal(0,0,math.pi)
        self.move(c,math.pi,True)
        rospy.sleep(5)


        #
        # self.car_model.CAR_RECOGNIZE = True
        #

        # b = self.make_move_base_goal(7.25,2.5,0)
        # self.move(b,0,True)
        # rospy.sleep(0.5)
        # self.turn_ang(-1,math.pi/2)
        # rospy.sleep(0.5)
        # self.adjust_pose(math.pi/2)
        # rospy.sleep(0.5)


        # self.has_arrive('/mnt/arrival.txt')
        # self.csv_writer.write()
        # self.is_go('/mnt/start_again.txt')
        # self.csv_writer.write()

        # self.turn_ang(1,math.pi)
        # rospy.sleep(1)

        # c = self.make_move_base_goal(7.25,0.45,math.pi)
        # self.move(c,math.pi,False)
        # rospy.sleep(1)


        # self.go_picture_point(2.13+bioas,0.45,0,'1-9')
        # self.go_picture_point(3.58+bioas,0.45,0,'1-10')
        # self.go_picture_point(4.78+bioas,0.45,0,'1-11')
        # self.go_picture_point(5.73+bioas,0.45,0,'1-12')
        # self.csv_writer.write()

        #
        # self.car_model.CAR_RECOGNIZE = True
        #
    
        # self.turn_ang(1,math.pi)
        # rospy.sleep(1)


        # destinition = self.make_move_base_goal(0.25,0.25,0)
        # self.move(destinition,math.pi/2,True)
        # self.turn_ang(-1,math.pi/2)
        # rospy.sleep(1)
        # self.adjust_pose(math.pi/2)
        # rospy.sleep(1)

        # self.has_arrive('/mnt/finish.txt')
        # self.csv_writer.write()

        # print("start to finish")
        # self.finish()
        # self.csv_writer.write()


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
        rospy.sleep(1)
        self.take_picture_need = True
        while self.image is None:
            continue
        rospy.sleep(1)

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
        self.car_model.append(image_name,f'/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/{image_name}.jpg')

        #self.car_model.append(image_name,corrected_image)
        self.take_picture_need = False

        time.sleep(0.5)
        self.image = None
        


    def initial_pose_for_amcl(self,x,y,z,yaw):
        initialpose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        initial_pose = PoseWithCovarianceStamped()

        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.header.frame_id = "map"
        
        initial_pose.pose.pose.position.x = x
        initial_pose.pose.pose.position.y = y
        initial_pose.pose.pose.position.z = z
        
        quaternion = quaternion_from_euler(0, 0, yaw) 
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
                        print(b,a,alpha)
                        continue
                    ab = b * math.cos(alpha)
                    print(alpha, ab)
                    if abs(alpha) > 0.01 and count < 3:
                        self.turn_ang(0.7,alpha)
                        count += 1
                        rospy.sleep(1)
                    else:
                        self.stop()
                        rospy.sleep(1)
                        print("靠墙成功！")
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


    def move(self, goal, target_yaw, is_xy_fix, timeout=30, retries=3):
        attempt = 0
        while attempt < retries:
            attempt += 1
            if attempt > 1:
                # try:
                #     laser_data = self.laser_data
                #     laser_ranges = np.array(laser_data.ranges)
                #     min_index = np.argmin(laser_ranges)
                #     rad = math.radians(min_index * laser_data.angle_increment)
                #     error = math.pi - rad
                #     if error > 0:
                #         self.turn_ang(0.8,abs(error))
                #     else:
                #         self.turn_ang(-0.8,abs(error))
                #     rospy.sleep(0.5)
                #     self.go_linear_x(0.1,0.1)
                #     rospy.sleep(0.5)
                # except Exception as e:
                #     pass
                rate = rospy.Rate(20)
                while not rospy.is_shutdown():
                    try:
                        (current_postion,rotation) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0)) 
                        rate.sleep()
                        break
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        continue
                _1, _2, yaw = euler_from_quaternion(rotation)
                self.initial_pose_for_amcl(current_postion[0],current_postion[1],0,yaw)

            self.move_base.send_goal(goal)
            success = self.move_base.wait_for_result(rospy.Duration(timeout))        
            if success:
                state = self.move_base.get_state()
                if state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Goal reached successfully on attempt %d", attempt)
                    rospy.sleep(0.5)
                    self.adjust_pose(target_yaw)
                    rospy.sleep(0.5)
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
                            self.turn_ang(0.8,math.pi/2)
                            rospy.sleep(0.5)
                            self.go_linear_x(0.1,current_position_y - goal.target_pose.pose.position.y)
                            rospy.sleep(0.5)
                            self.turn_ang(-0.8,math.pi/2)
                            rospy.sleep(1)
                        else:
                            self.turn_ang(-0.8,math.pi/2)
                            rospy.sleep(0.5)
                            self.go_linear_x(0.1,-(current_position_y - goal.target_pose.pose.position.y))
                            rospy.sleep(0.5)
                            self.turn_ang(0.8,math.pi/2)
                            rospy.sleep(1)
                    self.adjust_pose(target_yaw)
                    rospy.sleep(0.5)
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


    def please_takeoff(self,path):
        # hostname = '192.168.110.55'
        hostname = '192.168.31.66'
        username = 'forlinx'
        password = 'forlinx'
        filepath = path
        conn = None

        
        while True:
            try:
                conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})
                result = conn.run(f"touch {filepath}")

                if result.ok:
                    print("File_takeoff created successfully.")
                    conn.close()
                    break
                else:
                    print("File_takeoff created filed.Attempting to reconnect...")
                    if conn:
                        conn.close()
                    time.sleep(2)
                    continue

            except Exception as e:
                print("Attempting to reconnect...")
                if conn:
                    conn.close()
                time.sleep(2)


    def please_landing(self,path):
        # hostname = '192.168.110.55'
        hostname = '192.168.31.66'
        username = 'forlinx'
        password = 'forlinx'
        filepath = path
        conn = None

        
        while True:
            try:
                conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})
                result = conn.run(f"touch {filepath}")

                if result.ok:
                    print("File_landing created successfully.")
                    conn.close()
                    break
                else:
                    print("File_landing created filed.Attempting to reconnect...")
                    if conn:
                        conn.close()
                    time.sleep(2)
                    continue

            except Exception as e:
                print("Attempting to reconnect...")
                if conn:
                    conn.close()
                time.sleep(2)



    def is_go_charge(self,path, timeout_seconds=300):

        # hostname = '192.168.110.55'
        hostname = '192.168.31.66'
        username = 'forlinx'
        password = 'forlinx'
        filepath = path
        conn = None
        start_time = time.time()
        
        while True:
            try:
                conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})
                while True:
                    time.sleep(2)
                    if time.time() - start_time > timeout_seconds:
                        print("Timeout,car go charge!!!!!!!")
                        if conn:
                            conn.close()
                        return

                    result = conn.run('test -e {}'.format(filepath), hide=True)
                    if result.ok: 
                        print("landing succseefully")
                        conn.run(f'rm {filepath}')
                        conn.close()
                        return
                    else:
                        print("next check")
                        time.sleep(2)
            except Exception as e:
                print("Attempting to reconnect...")
                if conn:
                    conn.close()
                time.sleep(2)


    
    def finish(self):
        print("START TO FINISH:")
        while True:
            time.sleep(1)
            print("CHECK CAR RECOGNIZE COUNT:",self.car_model.pic_recognize_count)
            if self.car_model.pic_recognize_count==0:
                self.car_model.CAR_FINISH = True
                break
        print("CAR TASK FINISH")
        
        # while True:
        #     time.sleep(1)
        #     if self.plane_model.PIC_RECOGNIZE_COUNT==0:
        #         break
        
        while True:
            
            time.sleep(1)
            print("CHECK PLANE RECOGNIZE COUNT:",self.plane_model.pic_recognize_count)
            if self.plane_model.pic_recognize_count==0:
                self.plane_model.PLANE_FINISH = True
                break
            print("PLANE TASK FINISH")
                
        #self.plane_model_manager.join()
        #print("plane model manager join")
        #self.car_model_manager.join()
        #print("car model manager join")

        print("FINISH CAR RECOGNITION!")

        
        
        # mnt_write_path = f'/mnt/result{csv_time}.csv'
        

        # print('FINISH PLANE RECOGNITION!')
        print("START TO SAVE RESULT AS CSV FILE")
        # self.car_model.save()
        # self.plane_model.save()
        
        # try:
        #     self.csv_writer.write()
        #     print(f"SAVE AS /home/tta/Project_ws/src/rmep_nav/scripts/model_result/result{csv_time}.csv")
        # except Exception as ce:
        #     print(f"发生了一个错误在写入本地csv文件中")
        
        # try:
        #     self.csv_writer.write(write_path)
        #     print(f"SAVE AS {write_path}")
        # except Exception as ce:
        #     print(f"写入sd卡错误: {ce}")

        
        # try:

        #     hostname = '192.168.31.110'
        #     username = 'root'
        #     password = '123456'
        #     filepath = path

        #     conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})

        #     conn.run(f"touch {write_path}")
        # except e as ce:
        #     print( "传入飞机时发生错误")
            
        
        return True
    
def merge_csv_files(flight_file, car_file, output_file):
    # 读取result_flight.csv
    with open(flight_file, 'r', newline='', encoding='utf-8') as f_flight:
        flight_reader = csv.reader(f_flight)
        flight_data = list(flight_reader)
    
    # 读取result_car.csv
    with open(car_file, 'r', newline='', encoding='utf-8') as f_car:
        car_reader = csv.reader(f_car)
        car_data = list(car_reader)
    
    # 确保两个文件行数一致
    if len(flight_data) != len(car_data):
        raise ValueError("CSV files have different number of rows")
    
    # 创建合并后的数据
    merged_data = []
    
    # 处理标题行
    if flight_data and car_data:
        merged_data.append(flight_data[0])  # 使用任一文件的标题行
    
    # 合并数据行
    for i in range(1, len(flight_data)):
        flight_row = flight_data[i]
        car_row = car_data[i]
        
        # 确保列数一致
        if len(flight_row) != 4 or len(car_row) != 4:
            raise ValueError(f"Row {i} has inconsistent columns")
        
        # 处理I区水果种类：优先car，其次flight
        i_fruit = car_row[1] if car_row[1].strip() else flight_row[1]
        
        # 处理II区水果种类：优先car，其次flight
        ii_fruit = car_row[3] if car_row[3].strip() else flight_row[3]
        
        # 构建合并后的行（保留原始货架号格式）
        merged_row = [
            car_row[0],  # I区货架号（两个文件相同）
            i_fruit,
            car_row[2],  # II区货架号（两个文件相同）
            ii_fruit
        ]
        merged_data.append(merged_row)
    
    # 写入合并后的CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(merged_data)


if __name__ == '__main__':
    nav = Nav()
    rospy.spin()



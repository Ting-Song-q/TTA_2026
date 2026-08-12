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
import random

def num_to_fruit(num):
    fruit=None
    if(num==0):
        fruit="苹果"
    elif(num==1):
        fruit="香蕉"
    elif(num==2):
        fruit="橘子"
    elif(num==3):
        fruit="西瓜"
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
    img_height = 1088 
    img_width = 1440

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
    return xywh_list, cls_list, conf_list
    
# def filter_boxes(xywh_list, cls_list, conf_list, span_factor=1.25, layer=1, min_factor=0.2):
#     """
#     过滤和统计平面上的检测框
    
#     参数:
#         boxes: 包含检测框信息的对象，应有xywhn和cls属性
#         span_factor: 滑动窗口的扩展因子，默认为1.25
#         layer: 检测层级(1或2)，默认为1
#         min_factor: 最小面积因子，用于过滤小框，默认为0.2
    
#     返回:
#         tuple: (是否有效, 置信度最高的种类)
#     """
    
#     # 定义卡片的标准尺寸比例
#     CARD_SPAN_X = 4  # 宽度扩展倍数
#     CARD_SPAN_Y = 3  # 高度扩展倍数
    
#     # 定义框的宽高比范围
#     BOX_MIN_MORPH = 0.3  # 最小宽高比(1:1)
#     BOX_MAX_MORPH = 1.5  # 最大宽高比(3:2)
    
#     # 从检测结果中提取归一化坐标和类别

#     rbox_list = []  # 存储过滤后的框
#     rcls_list = []  # 存储过滤后的类别
#     rconf_list = [] # 存储过滤后的置信度
    
#     # 1. 如果没有识别到物体，返回False,并且随机选择一个种类
#     if len(cls_list) == 0:
#         random_cls = random.randint(0, 3)  
#         print("It seem to be a/an {}".format(num_to_fruit(random_cls)))
#         return False,random_cls 
    
#     # 2. 如果识别到物体
#     else:
#         BOX_VALID = False
        
#         BOX_XYWHN = 0
#         BOX_MAX_AREA = 0
        
#         CENTRAL_BIAS = 10  # 中心偏差阈值
#         CENTRAL_OBJECT_INDEX = 0  # 中心物体索引
        
#         # 1-1 过滤变形严重的识别框，宽高比不正常的都过滤
#         # 1-2 获取最大的识别框的面积
#         print("!!!!!!!")
#         print(xywh_list)
#         print(cls_list)
#         for obj_box, obj_cls, obj_conf in zip(xywh_list, cls_list, conf_list):
#             obj_width, obj_height = obj_box[2], obj_box[3]  #获取识别框的宽高
#             this_area = obj_width * obj_height  # 获取识别框的面积
#             if BOX_MAX_AREA < this_area: #获取最大的识别框的面积
#                 BOX_MAX_AREA = this_area
#             # 检查宽高比是否在合理范围内
#             if BOX_MIN_MORPH < (obj_width/obj_height) < BOX_MAX_MORPH:
#                 rbox_list.append(obj_box)
#                 rcls_list.append(obj_cls)
#                 rconf_list.append(obj_conf)
        
#         index = 0  # 当前处理框的索引
        
#         # 2-1 小框过滤(main)
#         # 2-2 上下比例窗口过滤
#         # 2-3 最近邻
#         for obj_box, obj_cls, obj_conf in zip(rbox_list, rcls_list, rconf_list):
#             obj_central_x, obj_central_y = obj_box[0], obj_box[1]
#             obj_width, obj_height = obj_box[2], obj_box[3]
            
#             print("TEST FOR RBOX_LIST:", rbox_list[0])
#             # 获取中心框的坐标和尺寸
#             centroid_x = rbox_list[CENTRAL_OBJECT_INDEX][0]
#             centroid_y = rbox_list[CENTRAL_OBJECT_INDEX][1]
#             centroid_w = rbox_list[CENTRAL_OBJECT_INDEX][2]
#             centroid_h = rbox_list[CENTRAL_OBJECT_INDEX][3]
            
#             # 检查框面积是否大于最小阈值
#             if not (obj_width*obj_height > BOX_MAX_AREA*min_factor):
#                 print("小框过滤未通过")
#             if obj_width*obj_height > BOX_MAX_AREA*min_factor:
#                 # 根据层级计算距离和检查窗口
#                 if layer == 1:  # 与底部(0.5,0)的距离
#                     my_distance = (obj_central_x-0.5)**2 + (obj_central_y)**2
#                     CHECK_WINDOW_XYWHN = [0.5, 0.25, centroid_w*14, centroid_h*10]
#                 elif layer == 2:    # 与顶部(0.5,1)的距离
#                     my_distance = (obj_central_x-0.5)**2 + (1-obj_central_y)**2
#                     CHECK_WINDOW_XYWHN = [0.5, 0.75, centroid_w*14, centroid_h*8]
#                 else:
#                     my_distance = (obj_central_x-0.5)**2 + (obj_central_y-0.5)**2
#                     CHECK_WINDOW_XYWHN = [0.5, 0.5, 0.8, 0.8]
                
#                 print("WINDOW:", CHECK_WINDOW_XYWHN)
#                 print("PICTURE:", rbox_list[CENTRAL_OBJECT_INDEX])
                
#                 # 检查框是否完全在窗口内
#                 if not (CHECK_WINDOW_XYWHN[0]-CHECK_WINDOW_XYWHN[2]/2 < obj_central_x < CHECK_WINDOW_XYWHN[0]+CHECK_WINDOW_XYWHN[2]/2):
#                     print("横向检测框未通过") # 说明横向的检测框超出了
#                 if not (CHECK_WINDOW_XYWHN[1]-CHECK_WINDOW_XYWHN[3]/2 < obj_central_y < CHECK_WINDOW_XYWHN[1]+CHECK_WINDOW_XYWHN[3]/2):
#                     print("纵向检测框未通过") #说明竖向检测框超出了
                
#                 # 如果框在窗口内且距离更近，则更新中心框
#                 if (my_distance < CENTRAL_BIAS) and \
#                    (CHECK_WINDOW_XYWHN[0]-CHECK_WINDOW_XYWHN[2]/2 < obj_central_x < CHECK_WINDOW_XYWHN[0]+CHECK_WINDOW_XYWHN[2]/2) and \
#                    (CHECK_WINDOW_XYWHN[1]-CHECK_WINDOW_XYWHN[3]/2 < obj_central_y < CHECK_WINDOW_XYWHN[1]+CHECK_WINDOW_XYWHN[3]/2):
#                     CENTRAL_BIAS = my_distance
#                     CENTRAL_OBJECT_INDEX = index
#                     BOX_VALID = True
                    
#             index += 1
        
#         # 3-1 滑动窗口
#         if BOX_VALID:
#             # 计算扩展后的滑动窗口坐标
#             centroid_x = rbox_list[CENTRAL_OBJECT_INDEX][0]
#             centroid_y = rbox_list[CENTRAL_OBJECT_INDEX][1]
#             centroid_w = rbox_list[CENTRAL_OBJECT_INDEX][2]
#             centroid_h = rbox_list[CENTRAL_OBJECT_INDEX][3]
            
#             BOX_XYWHN = [
#                 centroid_x,
#                 centroid_y,
#                 centroid_w * CARD_SPAN_X * span_factor,  # 扩展宽度
#                 centroid_h * CARD_SPAN_Y * span_factor   # 扩展高度
#             ]
            

#             MAX_CONF = 0
#             BEST_CLS = None

#             # 选选择滑动窗口内内置信度最高的框进行检测
#             for obj_box, obj_cls, obj_conf in zip(rbox_list, rcls_list, rconf_list):
#                 # 检查框是否在滑动窗口内
#                 if not (BOX_XYWHN[0]-BOX_XYWHN[2]/2 < obj_box[0] < BOX_XYWHN[0]+BOX_XYWHN[2]/2):
#                     print("滑动窗口横向未通过")
#                 if not (BOX_XYWHN[1]-BOX_XYWHN[3]/2 < obj_box[1] < BOX_XYWHN[1]+BOX_XYWHN[3]/2):
#                     print("滑动窗口纵向未通过")
                
#                 if (BOX_XYWHN[0]-BOX_XYWHN[2]/2 < obj_box[0] < BOX_XYWHN[0]+BOX_XYWHN[2]/2) and \
#                    (BOX_XYWHN[1]-BOX_XYWHN[3]/2 < obj_box[1] < BOX_XYWHN[1]+BOX_XYWHN[3]/2):
                    
#                     # 类别判断，选择置信度最高的一个类别，可能不需要但是保险起见
#                     if obj_conf > MAX_CONF:
#                         MAX_CONF = obj_conf
                        
#                         if -0.01 < obj_cls < 0.01:
#                             BEST_CLS = 0  # 苹果
#                         elif -0.01 < obj_cls - 1  < 0.01:
#                             BEST_CLS = 1  # 香蕉
#                         elif -0.01 < obj_cls - 2  < 0.01:
#                             BEST_CLS = 2  # 橘子
#                         elif -0.01 < obj_cls - 3  < 0.01:
#                             BEST_CLS = 3  # 西瓜
#                         else:
#                             print("I hope that I would never see this output")
            
#             print("I believe it is a/an {}".format(BEST_CLS))
#             return True, BEST_CLS
        
#         random_cls = random.randint(0, 3)  
#         print("It seem to be a/an {}".format(num_to_fruit(random_cls)))
#         return False,random_cls 


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
        random_cls = random.randint(0, 3)  
        print("It seem to be a/an {}".format(num_to_fruit(random_cls)))
        return False,random_cls 
    elif len(cls_list) == 1:
        print("I only saw one fruit")
        BEST_CLS = cls_list[0]
        print("-----")
        print(conf_list[0])
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
        print("----------")
        print(conf_list)

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
            

        print("I believe it is a/an {}".format(num_to_fruit(BEST_CLS)))
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
        self.device = device
        self.pic_recognize_lock = threading.Lock()  # 图片计数线程锁
        self.pic_recognize_count = pic_recognize_count  # 剩余待识别图片计数器

        # 模型相关路径
        self.model_path = model_path
        self.model_input_path = model_input_path
        
        # 任务队列：输入队列存储待识别图片，输出队列存储识别结果
        self.model_input_buffer = Queue()
        self.model_output_buffer = Queue()

        # 线程同步机制
        self.condition = threading.Condition()  # 条件变量控制任务调度
        self.threads_pool = []  # 工作线程池
        
        # 初始化工作线程 (守护线程)
        for i in range(0, num_workers):  # 创建4个工作线程
            worker = Thread(target=self.work)  # 设置线程目标为work方法
            worker.daemon = True  # 主线程退出时自动终止
            worker.start()  # 启动线程
            self.threads_pool.append(worker)
            
        # 任务状态标志 (飞机专用)
        self.PLANE_1 = False     # 标记第1批图片（1张）是否已处理
        self.PLANE_2 = False     # 标记第2批图片（1张）是否已处理
        self.PLANE_FINISH = False  # 标记飞机任务是否全部完成

    def plane_run(self):
        """飞机设备的任务调度循环"""
        while True:
            time.sleep(3)  # 每3秒检查一次
            check = self.plane_task()  # 检查新任务
            
            # 根据任务状态通知工作线程
            if check == 1:  # 第一批图片就绪
                with self.condition:
                    self.condition.notify_all()  # 唤醒所有工作线程
            
            if check == 2:  # 第二批图片就绪
                with self.condition:
                    self.condition.notify_all()
            
            # 终止条件：所有批次完成
            if self.PLANE_FINISH:
                print("finish plane task")
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
            match = re.search(r'_(\d+)-(\d+)\.jpg$', model_input)
            if match:
                var1 = match.group(1)  # 批次ID (未使用)
                var2 = match.group(2)  # 层级ID
                
                # 根据层级选择处理逻辑
                if -0.1 < float(var2) < 4.1:
                    return filter_boxes(res_xywh,res_cls,res_conf,layer=1)  # 选择与中部最近的
                elif 4.1 < float(var2) < 8.1:
                    return filter_boxes(res_xywh,res_cls,res_conf,layer=2)  # 选择与中间最近的
                
        except Exception as e:
            print(f"Error in recognize: {str(e)}")
            return 0, 0  # 返回默认值

    def plane_task(self):
        """
        飞机设备任务调度：监控目录信号文件并加载对应批次的图片
        
        返回:
            int: 1(第一批加载), 2(第二批加载), False(无新任务)
        """
        file_names = os.listdir(self.model_input_path)
        
        # 检测第一批完成信号 (finish4.txt)
        if 'finish4.txt' in file_names and not self.PLANE_1:
            print("FINISH 4 SIGNAL!!!")
            # 加载第一批图片 (1440x1088_1*.jpg)
            img_batch1 = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_1*')))
            for img_path in img_batch1:
                self.append(img_path, img_path)  # 添加到输入队列
            self.PLANE_1 = True  # 更新状态
            return 1
        
        # 检测第二批完成信号 (finish8.txt)
        elif 'finish8.txt' in file_names and not self.PLANE_2:
            print("FINISH 8 SIGNAL!!!")
            # 加载第二批图片 (1440x1088_2*.jpg)
            img_batch2 = sorted(glob.glob(os.path.join(self.model_input_path, '1440x1088_2*')))
            for img_path in img_batch2:
                self.append(img_path, img_path)
            self.PLANE_2 = True
            return 2
        
        return False

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
    
class CsvWriter:
    def __init__(self, output_file_path):
        self.output_file_path = output_file_path
        self.fruit_1 = None
        self.fruit_2 = None

        # 数字到字符串的映射
        self._fruit_map = {
            0: "apple",
            1: "banana",
            2: "orange",
            3: "watermelon",
            4: "other"
        }

        with open(output_file_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(('I区货架号', '水果种类', 'II区货架号', '水果种类'))

            for i in range(9, 13):
                fruit1 = self._fruit_map[random.choice((0, 1, 2, 3, 4))]
                fruit2 = self._fruit_map[random.choice((0, 1, 2, 3, 4))]
                row = (f'="1-{i}"', fruit1, f'="2-{i}"', fruit2)
                writer.writerow(row)
                print(f"row{i}: {row}")

        print("init over")
    

class Nav:
    def __init__(self):
        rospy.init_node('nav', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_subscriber = rospy.Subscriber('/scan', LaserScan, self.laser_callback,queue_size=1)
        # self.image_raw = rospy.Subscriber("ep_cam/image_raw", Image, self.camera_cb, queue_size=1)
        # self.cv_bridge = CvBridge()
        # self.laser_data = None
        # self.image = None
        # self.take_picture_need = False
        # self.image_lock = threading.Lock()

        # self.plane_model = Model('plane',
        #     "/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-2.pt",
        # model_input_path='/home/tta/Project_ws/src/rmep_nav/scripts/image_from_flight/',
        # pic_recognize_count=16)

        # self.car_model = Model('car',
        #     "/home/tta/Project_ws/src/rmep_nav/scripts/model/720car-1.pt",
        # model_input_path = '/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/',
        # pic_recognize_count=8)
        
        # self.car_model_manager = Thread(target=self.car_model.car_run)
        # self.car_model_manager.start()
        # self.plane_model_manager = Thread(target=self.plane_model.plane_run)
        # self.plane_model_manager.start()
        # self.csv_time = rospy.Time.now()
        # self.csv_writer = CsvWriter(self.car_model.model_output_buffer,
        #                     self.plane_model.model_output_buffer,
        #                     f'/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result{self.csv_time}.csv')
    
        #########################
        self.laser_data = None
        # self.plane_model = Model('plane',
        #     "/home/tta/Project_ws/src/rmep_nav/src/yolov5/best_0714.pt",
        # model_input_path='/home/tta/Project_ws/src/rmep_nav/scripts/image_from_flight/',
        # pic_recognize_count=2)

        # self.plane_model_manager = Thread(target=self.plane_model.plane_run)
        # self.plane_model_manager.start()
        self.csv_time = rospy.Time.now()
        self.csv_writer = CsvWriter(f'/home/tta/Project_ws/src/rmep_nav/scripts/model_result/result_car.csv')
        #########################


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
        #rospy.sleep(60)
        # def move_to_initial_position(self, goal_left, goal_back)
        self.move_to_initial_position(2.3, 0.9)
        self.move_to_initial_position(2.3, 0.9)

        # self.is_go('/mnt/start.txt')
        print("isgo!!!!!!!!!!!!!!")
        speed=0.5

        self.go_linear_x(speed,2.4)
        rospy.sleep(1)

        self.go_linear_y(-speed,0.85)
        rospy.sleep(1)
        #self.align_wall()
        #self.align_wall()
        rospy.sleep(0.5)
        #self.move_adjust_goal_position_lb(1.4,3.4)
        #self.move_adjust_goal_position_lb(1.4,3.4)
        rospy.sleep(1)
        #self.align_wall()
        #self.align_wall()
        rospy.sleep(0.5)

        # #小车到达1区，给飞机发送指令起飞
        self.please_takeoff('/home/forlinx/catkin_ws/src/tta_m3e_rtsp/my_msg/takeoff.txt')
        print("the car has arrived the first goal and the plane can take off!!!!!!!!!!!!!!")
        # #飞机检索完1区，给小车回复启动
        #self.is_go_2('/mnt/go_2.txt')
        rospy.sleep(25)
        #rospy.sleep(5)

        print("1区驶往2区.......")

        self.go_linear_x(speed,2.25)
        rospy.sleep(1)
        self.go_linear_y(speed,1.70)
        #self.move_adjust_goal_position_rf(1.4,3.1)
        #self.move_adjust_goal_position_rf(1.4,3.1)
        rospy.sleep(1)       

        # #飞机检索完2区，并已经成功降落，给小车回复启动
        self.is_go_charge('/home/forlinx/catkin_ws/src/tta_m3e_rtsp/my_msg/go_charge.txt',480)
        print("2区驶往充电区......")

        self.go_linear_y(-speed,0.85)
        rospy.sleep(1)
        self.go_linear_x(speed,2)
        rospy.sleep(1)
        self.align_wall_dengyao()
        self.align_wall_dengyao()
        rospy.sleep(0.5)   
        self.move_adjust_goal_position_rf(2.3,1.1)
        self.move_adjust_goal_position_rf(2.3,1.1)
        self.align_wall_dengyao()
        self.align_wall_dengyao()
        rospy.sleep(0.5)
        
        #充电区充电
        print("充电中......")
        rospy.sleep(1)   

        #返航
        print("返航......") 
        self.go_linear_x(-speed,7)    
        
        self.finish()   

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
        # self.take_picture(image_name)
    

    # def camera_cb(self, msg):
    #     with self.image_lock:
    #         if self.take_picture_need:
    #             self.image = self.cv_bridge.imgmsg_to_cv2(msg,"bgr8")
    #         else:
    #             return
        
    # def take_picture(self, image_name):
    #     self.take_picture_need = True
    #     while self.image is None:
    #         continue
    #     rospy.sleep(1)

    #     image = copy.deepcopy(self.image)
    #     # image = self.image
    #     # / 有无必要使用deepcopy？
    #     dist_coeffs = np.array([-0.0444954569665007, -0.00201770323876866, 0, 0, -0.0159863040899372])
    #     intrinsic = np.array([[624.302683276108, 0, 632.376721286788],
    #                     [0, 623.915644264347, 371.420934824104],
    #                     [0, 0, 1]])
    #     corrected_image = cv2.undistort(image, intrinsic, dist_coeffs)     
    #     # 可以注释掉的deepcopy，理论上是为了防止脏写，但是OS或cv2包可能已经进行了维护   
    #     #
    #     cv2.imwrite(f'/home/tta/Project_ws/src/rmep_nav/scripts/image_from_car/{image_name}.jpg',corrected_image)
    #     # 

    #     self.car_model.append(image_name,corrected_image)
    #     self.take_picture_need = False

    #     time.sleep(0.5)
    #     self.image = None
        
    def download_picture(remote_path,local_path):

        # 配置信息
        hostname = '192.168.3.58'
        username = 'root'
        password = '123456'
        remote_file = remote_path  # 远程文件完整路径
        local_file = local_path     # 本地保存完整路径


        start_time = time.time()
        timeout = 20  # 总超时时间（秒）
        connection = None

        
        try:
            while time.time() - start_time < timeout:
                try:
                    # 建立SFTP连接
                    sftp = pysftp.Connection(
                        host=hostname,
                        username=username,
                        password=password,
                        cnopts=pysftp.CnOpts(),  # 禁用主机密钥检查
                        timeout=3
                    )
                    print(f"连接成功! 耗时: {time.time()-start_time:.2f}秒")
                    
                except Exception as e:
                    print(f"连接尝试失败: {str(e)}")
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0: 
                        break
                    time.sleep(1)  # 失败后等待1秒重试          
                                
            local_dir = os.path.dirname(local_file)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)

            # 直接下载文件
            print(f"正在下载: {remote_file} -> {local_path}")
            sftp.get(remote_file, local_file)
            print("文件下载完成!")
        
        finally:
            if connection:
                connection.close()
        
        print("10秒超时，跳过下载操作")




        
    def please_takeoff(self,path):
        # hostname = '192.168.110.55'
        hostname = '192.168.3.58'
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


    def is_go_charge(self,path, timeout_seconds=300):

        # hostname = '192.168.110.55'
        hostname = '192.168.3.58'
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

    def is_go_charge(self,path, timeout_seconds=480):

            # hostname = '192.168.110.55'
            hostname = '192.168.3.58'
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
        #self.align_wall()
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
        #self.align_wall()
        laser_data = self.laser_data
        left_distance = self.get_distance(laser_data,-90)
        back_distance = self.get_distance(laser_data,0)
        print(left_distance,back_distance)

    def move_adjust_goal_position_lb(self, goal_left, goal_back):
        #self.align_wall()
        retries_y = 0
        retries_x = 0
        backoff_step=0.05
        while not rospy.is_shutdown():
            if self.laser_data:
                laser_data = self.laser_data
                left_distance = self.get_distance(laser_data,-90)
                print("现在与左侧的距离为\n",left_distance)
                back_distance = self.get_distance(laser_data,0)
                if left_distance == float('inf') or back_distance == float('inf'):
                    continue
                # ---- 2. 障碍物检测与处理 ----
                obstacle_flag = left_distance < 0.8 
                if obstacle_flag:
                    rospy.logwarn("Obstacle detected (<0.8 m). Backing off 5 cm...")
                # 后退 5 cm
                    self.go_linear_x(-0.1, backoff_step)
                    rospy.sleep(0.5)

                # 重新测距
                    laser_data   = self.laser_data
                    left_distance = self.get_distance(laser_data, -90)

                    still_too_close = left_distance < 0.8 
                    if still_too_close:
                        rospy.logerr("Still too close after backoff. Aborting adjustment.")
                    # 前进 5 cm 回到原位
                        self.go_linear_x(0.1, backoff_step)
                        rospy.sleep(0.5)
                        return              # 彻底打断本次修正
                    else:
                        rospy.loginfo("Backoff successful, continue fine-tuning.")
                    # 继续 while 循环做正常微调
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
        #self.align_wall()
        laser_data = self.laser_data
        left_distance = self.get_distance(laser_data,-90)
        back_distance = self.get_distance(laser_data,0)
        print(left_distance,back_distance)










    def move_adjust_goal_position_rf(self, goal_right, goal_front):
        #self.align_wall_l()
        retries_y = 0
        retries_x = 0
        backoff_step=0.05
        while not rospy.is_shutdown():
            if self.laser_data:
                laser_data = self.laser_data
                #left_distance = self.get_distance(laser_data,-90)
                right_distance=self.get_distance(laser_data,90)
                print("现在距离右侧的距离为：\n",right_distance)
                #back_distance = self.get_distance(laser_data,0)
                front_distance=self.get_distance(laser_data,180)
                if right_distance == float('inf') or front_distance== float('inf'):
                    continue
                # ---- 2. 障碍物检测与处理 ----
                obstacle_flag = right_distance < 0.8 
                if obstacle_flag:
                    rospy.logwarn("Obstacle detected (<0.8 m). Backing off 5 cm...")
                # 后退 5 cm
                    self.go_linear_x(-0.1, backoff_step)
                    rospy.sleep(0.5)

                # 重新测距
                    laser_data   = self.laser_data
                    right_distance = self.get_distance(laser_data, 90)
                    front_distance = self.get_distance(laser_data, 180)

                    still_too_close = right_distance < 0.8 
                    if still_too_close:
                        rospy.logerr("Still too close after backoff. Aborting adjustment.")
                    # 前进 5 cm 回到原位
                        self.go_linear_x(0.1, backoff_step)
                        rospy.sleep(0.5)
                        return              # 彻底打断本次修正
                    else:
                        rospy.loginfo("Backoff successful, continue fine-tuning.")
                    # 继续 while 循环做正常微调
                        continue

                if abs(right_distance - goal_right) > 0.02 and retries_y < 3:
                    error_y = abs(right_distance - goal_right)
                    speed =  0.1 * (right_distance - goal_right) / abs(right_distance - goal_right)  
                    self.go_linear_y(speed,error_y)
                    retries_y += 1
                    rospy.sleep(1)
                    continue
                if abs(goal_front - front_distance) > 0.02 and retries_x < 3:
                    error_x = abs(goal_front - front_distance)
                    speed = -0.1 * (goal_front - front_distance) / abs(goal_front - front_distance)
                    self.go_linear_x(speed,error_x)
                    retries_x += 1
                    rospy.sleep(1)
                    continue
                # self.stop()
                break
            self.rate.sleep()
        #self.align_wall_l()
        laser_data = self.laser_data
        right_distance = self.get_distance(laser_data,90)
        front_distance = self.get_distance(laser_data,180)
        print(right_distance,front_distance)


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
                    THETA = math.pi / 180 * 5
                    b = self.get_distance(laser_data, 0)
                    a = self.get_distance(laser_data, 5)
                    alpha = math.atan((a * math.cos(THETA) - b) / (a * math.sin(THETA)))
                    if b == float('inf') or a == float('inf') or alpha == np.nan:
                        print(b,a,alpha)
                        continue
                    ab = b * math.cos(alpha)
                    print(alpha, ab)
                    if abs(alpha) > 0.01 and count < 3:
                        self.turn_ang(0.35,alpha)
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

    def align_wall_l(self):
        count = 0
        while not rospy.is_shutdown():
            if self.laser_data:
                try:
                    laser_data = self.laser_data
                    THETA = math.pi / 180 * 5
                    b = self.get_distance(laser_data, -180)
                    a = self.get_distance(laser_data, -175)
                    alpha = math.atan((a * math.cos(THETA) - b) / (a * math.sin(THETA)))
                    if b == float('inf') or a == float('inf') or alpha == np.nan:
                        print(b,a,alpha)
                        continue
                    ab = b * math.cos(alpha)
                    print(alpha, ab)
                    if abs(alpha) > 0.01 and count < 3:
                        self.turn_ang(0.35,alpha)
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

    
    

    def align_wall_dengyao(self):
        count = 0
        while not rospy.is_shutdown():
            if self.laser_data:
                try:
                    laser_data = self.laser_data
                    THETA = math.pi / 180 * 30
                    a = self.get_distance(laser_data, -150)
                    b = self.get_distance(laser_data, 150)

                    if a >= b:
                        s1 = a - b
                        s2 = 2 * a * math.sin(THETA)
                        
                        alpha = math.asin(( s1 * math.sin(90 - THETA) / s2 ) / ( 1 + s1 / s2))
                        
                    else:
                        s1 = b - a
                        s2 = 2 * b * math.sin(THETA)

                        alpha = -math.asin(( s1 * math.sin(90 - THETA) / s2 ) / ( 1 + s1 / s2))

                    if b == float('inf') or a == float('inf') or alpha == np.nan:
                        print(b,a,alpha)
                        continue

                    print("!!!!!{}".format(alpha))
                    if abs(alpha) > 0.01 and count < 3:
                        self.turn_ang(0.35,alpha)
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


    
    def finish(self):
        print("START TO FINISH:")
        
        command = f"python /home/tta/Project_ws/src/rmep_nav/scripts/finish_sw.py"
        # 调用目标程序并传递参数
        return os.system(command)

if __name__ == '__main__':
    nav = Nav()
    rospy.spin()



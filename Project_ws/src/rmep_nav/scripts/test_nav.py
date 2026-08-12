#!/usr/bin/python3
# coding=UTF-8

import cv2
import rospy
import sys
import numpy as np
from pyzbar import pyzbar
from fabric2 import Connection
from std_msgs.msg import Bool
import inter


from sensor_msgs.msg import Image
from cv_bridge import CvBridge , CvBridgeError
import math
from geometry_msgs.msg import Pose, Point, Quaternion, Twist

class test:

    def __init__(self):
        self.QR_code =np.zeros(0)
        self.cv_bridge = CvBridge()
        self.camera_info_sub  = rospy.Subscriber("ep_cam/image_raw", Image, self.camera_info_cb, queue_size=30)
        self.test1()


    def go_linear(linear_speed,goal_distance):
        cmd_vel = cmd_vel = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        target_linear_time = math.fabs(goal_distance /linear_speed)
        twist_linear = Twist()
        twist_linear.linear.x = linear_speed
        twist_linear.linear.y = 0
        twist_linear.linear.z = 0
        twist_linear.angular.x = 0
        twist_linear.angular.y = 0
        twist_linear.angular.z = 0
        go_start_time = rospy.Time.now()
        rate = rospy.Rate(50)  
        while (rospy.Time.now() - go_start_time).to_sec() < target_linear_time:
            cmd_vel.publish(twist_linear)
            rate.sleep()
        twist_linear.linear.x = 0
        cmd_vel.publish(twist_linear)

    def turn_ang(ang_speed,goal_rotation):
        cmd_vel = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        twist_ang = Twist()
        twist_ang.angular.z=ang_speed
        twist_ang.linear.x = 0
        twist_ang.linear.y = 0
        twist_ang.linear.z = 0
        twist_ang.angular.x = 0
        twist_ang.angular.y = 0
        target_rotation_time = math.fabs(goal_rotation/ang_speed)
        turn_start_time = rospy.Time.now()
        rate = rospy.Rate(50)  
        while (rospy.Time.now() - turn_start_time).to_sec() < target_rotation_time:
            cmd_vel.publish(twist_ang)
            rate.sleep()
        twist_ang.angular.z = 0
        cmd_vel.publish(twist_ang)

    def is_go(path):
        hostname = '192.168.110.55'
        username = 'root'
        password = '123456'
        filepath = path

        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})

        while True:
            try:
                result = conn.run('test -e {}'.format(filepath), hide=True)
                if result.ok: 
                    conn.run(f'rm {filepath}')
                    conn.close()
                    break
            except Exception as e:
                print("continue to check")
                continue
        
    def has_arrive(path):
        hostname = '192.168.110.55'
        username = 'root'
        password = '123456'
        filepath = path

        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})

        conn.run(f"touch {filepath}")

    def camera_info_cb(self,msg):
            

            image = self.cv_bridge.imgmsg_to_cv2(msg,"bgr8")
            #print(image)
            ImgCode = self.QR_Scan(image)

            for temp in range(0,max(0,np.size(ImgCode))):
                Judge = True
                #print("test1")

                for real in range(0,max(0,np.size(self.QR_code))):
                    if(ImgCode[temp]==self.QR_code[real]):
                        #print("test2")
                        Judge = False


                if(Judge):
                    #print("test3")
                    self.QR_code = np.append(self.QR_code,ImgCode[temp])

                    print(self.QR_code)

            if(np.size(self.QR_code)!=0):
                pass
                #print(self.QR_code)

            cv2.imshow("Robot",image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
            
    def test1(self):
        rospy.sleep(10)
        hostname = '192.168.110.55'
        username = 'root'
        password = '123456'

        filepath = '/mnt/tta/list_of_goods_fromCAR.txt'
        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})
        print("connect suc!!")
        b = inter.create_file(conn,filepath)
        print(b)
        print("test_opr")
        count = 0   
        a = inter.file_echo(conn,filepath,"11111")
        while count<np.size(self.QR_code):
            print(self.QR_code[count])

            a = inter.file_echo(conn,filepath,str(self.QR_code[count]))
            count+=1  
        conn.close() 

    def QR_Scan(self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        decoded = pyzbar.decode(gray)
        Code = np.zeros(0)
        for barcode in decoded:
            (x,y,w,h) = barcode.rect
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            Data = barcode.data.decode("utf-8")
            Type = barcode.type
            codetext = "{}".format(Data)
            Code = np.append(Code,codetext)
        return Code



rospy.init_node('test_nav', anonymous=True)
a = test()
rospy.spin()

#is_go('/mnt/start.txt')



#has_arrive('/mnt/arrival.txt')
#is_go('/mnt/start_again.txt')
'''turn_ang(-0.5,1.57079633)
go_linear(0.2,0.25)
turn_ang(-0.5,1.57079633)
go_linear(0.2,1)
turn_ang(-0.5,1.57079633)
go_linear(0.2,0.25)
has_arrive('/mnt/finish.txt') '''


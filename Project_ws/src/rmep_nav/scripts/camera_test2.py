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

class img_processing:
    
    global frame_cnt    
    frame_cnt=1
    

    def __init__(self):
        self.QR_code =np.zeros(0)
        self.cv_bridge = CvBridge()
        self.camera_info_sub  = rospy.Subscriber("ep_cam/image_raw", Image, self.camera_info_cb, queue_size=30)
    

    def QR_Scan(self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        decoded = pyzbar.decode(gray)
        Code = np.zeros(0)
        for barcode in decoded:
            (x,y,w,h) = barcode.rect
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            Data = barcode.data.decode("utf-8")
            Type = barcode.type
            codetext = "{}({})".format(Data,Type)
            Code = np.append(Code,codetext)
        return Code
 
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



if __name__ == '__main__':
    
    rospy.init_node('img_processing', anonymous=True)
    img_processing = img_processing()
    rospy.spin()
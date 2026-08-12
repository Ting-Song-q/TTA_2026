#!/usr/bin/python3
# coding=UTF-8

import socket
import time
import cv2
import os
import numpy as np
import threading


def receive_images(client_socket, save_folder):
    while True:
        time.sleep(0.2)
        try:
            # 接收图像名称
            image_name = client_socket.recv(50)
            # if not image_name:
            #     break
            while True:
                time.sleep(0.2)
                if len(image_name) == 50:
                    image_name = image_name.decode('utf-8').strip()  # 解码并移除末尾空格
                    # if image_name == "finish1":
                    #     txt_path = os.path.join(save_folder,"finish1.txt")
                    #     with open(txt_path,'w'):
                    #         pass
                    # if image_name == "finish2":
                    #     txt_path = os.path.join(save_folder,"finish2.txt")
                    #     with open(txt_path,'w'):
                    #         pass
                    # if image_name == "finish3":
                    #     txt_path = os.path.join(save_folder,"finish3.txt")
                    #     with open(txt_path,'w'):
                    #         pass
                    if image_name == "finish4":
                        txt_path = os.path.join(save_folder,"finish4.txt")
                        with open(txt_path,'w'):
                            pass
                    # if image_name == "finish5":
                    #     txt_path = os.path.join(save_folder,"finish5.txt")
                    #     with open(txt_path,'w'):
                    #         pass
                    # if image_name == "finish6":
                    #     txt_path = os.path.join(save_folder,"finish6.txt")
                    #     with open(txt_path,'w'):
                    #         pass
                    # if image_name == "finish7":
                    #     txt_path = os.path.join(save_folder,"finish7.txt")
                    #     with open(txt_path,'w'):
                    #         pass
                    if image_name == "finish8":
                        txt_path = os.path.join(save_folder,"finish8.txt")
                        with open(txt_path,'w'):
                            pass
                    client_socket.send('A'.encode())
                    break
                else:
                    client_socket.send('N'.encode())
                    image_name = client_socket.recv(50)
            
            if (image_name == "finish4") or (image_name == "finish8"):
                print("此次检查结束")
                break 

            # 接收图像大小
            size_data = client_socket.recv(8)
            while True:
                time.sleep(0.2)
                if len(size_data) == 8:
                    size = int(size_data.decode().strip())
                    client_socket.send('A'.encode())
                    print("检查图像大小，大小为：" + str(size))
                    break
                else:
                    client_socket.send('N'.encode())
                    size_data = client_socket.recv(8)

            if size == -1:
                # 收到特殊标记，表示无法读取图像文件
                continue

            # 接收图像数据
            image_data = b''
            while len(image_data) < size:
                data = client_socket.recv(1024)
                if not data:
                    break
                image_data += data

            # 判断是否正确接收到图像
            if len(image_data) < size:
                client_socket.send('N'.encode())
                continue
            
            # 将接收到的数据解码成图像
            image_np = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            # 保存图像到文件
            image_filename = f"{image_name}"
            image_path = os.path.join(save_folder, image_filename)
            cv2.imwrite(image_path, image_np)
            print(f"检查图像正确: {image_filename}")
            
            # 发送接收确认
            client_socket.send('A'.encode())
            print("图像检查完成")

        except ConnectionError as ce:
            print(f"连接错误: {ce}")
            break


def client_handler(client_socket, save_folder):
    receive_images(client_socket, save_folder)
    client_socket.close()


def server_program(save_folder):
    ip = '192.168.31.89'  # xiaomi
    # ip = '192.168.110.128'  # aybt
    port = 6000
    server_socket = None
    # 第一个while循环确保服务端正确创建并保持一直打开
    while True:
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((ip, port))
            server_socket.listen(20)
            print("等待连接···")

            # 第二个循环保证连接的建立
            while True:
                try:
                    client_socket, addr = server_socket.accept()
                    print("连接来自: " + str(addr))
                    # 启动线程执行任务
                    thread = threading.Thread(target=client_handler, args=(client_socket, save_folder))
                    thread.start()
                except socket.error as e:
                    print(f"接受客户端连接时发生错误: {e}")
                    time.sleep(2)
                    continue
        except socket.error as e:
            print(f"服务端创建异常: {e}")
            time.sleep(2)
            continue

        finally:
            if server_socket:
                server_socket.close()
                print("服务器套接字已关闭")


if __name__ == '__main__':
    save_folder = '/home/tta/Project_ws/src/rmep_nav/scripts/image_from_flight_hand'
    # save_folder = 'D:\\socket\\server_pictures'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    try:
        server_program(save_folder)
    except KeyboardInterrupt:
        print("服务器正在关闭...")

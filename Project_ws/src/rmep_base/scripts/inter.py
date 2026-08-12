#! /usr/bin/env python3


import time
from venv import create
import rospy
from fabric2 import Connection



def create_file(conn, filepath):
    res = conn.run(f"touch {filepath}")
    return res

def check_file(conn,filepath):

    while True:
        try:
            result = conn.run('test -e {}'.format(filepath), hide=True)
            if result.ok:
                # Æô¶¯
                break
        except Exception as e:
            print("½øÐÐÏÂÒ»´Î²éÑ¯")
            continue
       


def delete_file(conn,filepath):
    res = conn.run(f'rm -f {filepath}')
    return res

def file_operation():
    hostname = '192.168.110.55'
    username = 'root'
    password = '123456'
    filepath = '/mnt/tta/test.txt'

    #½¨Á¢Á¬½Ó
    conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})


    create_file(conn,filepath)
    #ÂÖÑ¯¼ì²éÎÄ¼þÊÇ·ñ´æÔÚ£¬Èç¹û´æÔÚÔòÆô¶¯
    check_file(conn,filepath)
    print("²éÑ¯")

    # Æô¶¯³É¹¦ºóÉ¾³ýÎÄ¼þ
    # while True:
            
    file_echo(conn,filepath,"tta_ooo")
    delete_file(conn,filepath)
    # time.sleep(0.5)
    #print("susc del")

    # time.sleep(1)
    #create_file(conn,filepath)
    #create_file(conn,filepath)
    file_echo(conn,filepath,"tta_linker")  
    file_echo(conn,filepath,"getgetget")
    #delete_file(conn,filepath)  
    #file_echo()
    # time.sleep(2)
    conn.close()


def file_echo(conn,filepath,echo_param):
    #filepath = '/mnt/tta/test.txt'
    command = "echo "+str(echo_param)+" >> "+ str(filepath)
    res=conn.run(command)
    return res
    
    

if __name__ == '__main__':
 

    file_operation()
    
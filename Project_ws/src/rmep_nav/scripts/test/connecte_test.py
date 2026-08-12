#!/usr/bin/python3
# coding=UTF-8


from fabric2 import Connection
import time


def has_arrive(path):
        hostname = '192.168.110.55'
        # hostname = '192.168.31.110'
        username = 'root'
        password = '123456'
        filepath = path
        conn = None

        
        while True:
            try:
                conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})
                result = conn.run(f"touch {filepath}")

                if result.ok:
                    print("File created successfully.")
                    conn.close()
                    break
                else:
                    print("File created filed.Attempting to reconnect...")
                    if conn:
                        conn.close()
                    time.sleep(1)
                    continue
                    
            except Exception as e:
                print("Attempting to reconnect...")
                if conn:
                    conn.close()
                time.sleep(1)


def is_go(path):

        hostname = '192.168.110.55'
        # hostname = '192.168.31.110'
        username = 'root'
        password = '123456'
        filepath = path
        conn = None
        
        while True:
            try:
                conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})
                while True:
                    time.sleep(0.5)
                    result = conn.run('test -e {}'.format(filepath), hide=True)
                    if result.ok: 
                        print("fly succseefully")
                        conn.run(f'rm {filepath}')
                        conn.close()
                        return
                    else:
                        print("next check")
                        time.sleep(0.5)
            except Exception as e:
                print("Attempting to reconnect...")
                if conn:
                    conn.close()
                time.sleep(2)


if __name__ == "__main__":
    is_go("/mnt/start.txt")
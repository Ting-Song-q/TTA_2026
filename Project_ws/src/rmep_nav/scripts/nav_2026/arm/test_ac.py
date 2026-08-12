# mock_stm32_win.py  —— 在「假 STM32」那台 Win 上运行
import serial
import time

PORT = "COM8"      # 改成那台电脑的串口号
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=0.1)
print("假 STM32 监听 {} @ {}".format(PORT, BAUD))

buf = b""
try:
    while True:
        n = ser.in_waiting
        if n:
            chunk = ser.read(n)
            buf += chunk
            print("RX: {!r}  hex={}".format(chunk, chunk.hex()))
            # 收到 $GETA! 就回 AAA（与 test_win 判定一致）
            if b"$GETA!" in buf:
                ser.write(b"AAA")
                ser.flush()
                print("TX: AAA")
                buf = b""
            # 其它指令也可简单回 AAA，方便 --diagnose
            elif b"!" in buf:
                ser.write(b"AAA")
                ser.flush()
                print("TX: AAA (echo)")
                buf = b""
        else:
            time.sleep(0.02)
except KeyboardInterrupt:
    pass
finally:
    ser.close()
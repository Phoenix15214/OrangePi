import os
os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts/truetype/dejavu')

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import process_lib.control_lib as ctrl
from multiprocessing import Process, Pipe, shared_memory, Value
from threading import Thread
import time
from detect import main as detect_main
from track import main as track_main
from transmit import Send_Process

shm_name = 'shared_frame'
pipe1, pipe2 = Pipe()

def main():
    frame_ready = Value('b', False)
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=640*480*3)
    shm.close()
    p_track = Process(target=track_main, args=(shm_name, frame_ready, pipe2))
    p_detect = Process(target=detect_main, args=(shm_name, frame_ready, pipe2))
    p_transmission = Process(target=Send_Process, args=(pipe1, "justfloat"))

    p_transmission.start()
    p_track.start()
    p_detect.start()
    

    p_track.join()
    p_detect.join()
    p_transmission.join()

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.unlink()  # 删除共享内存段
    except FileNotFoundError as e:
        print(f"共享内存段已被删除: {e}")
    except Exception as e:
        print(f"删除共享内存段时发生错误: {e}")

if __name__ == '__main__':
    main()
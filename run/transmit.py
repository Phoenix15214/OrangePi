import socket
from multiprocessing import Process, Pipe
from threading import Thread
import process_lib.control_lib as ctrl
import struct

message = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
server_socket = None
isConnected = False
pack = ctrl.SerialPacket(port="/dev/ttyUSB0", baudrate=115200, timeout=0.1)

def Parse_Input(msg):
    start_flag = ":"
    end_flag = "\n"
    start_pos = msg.find(start_flag)
    content_pos = start_pos + len(start_flag)
    end_pos = msg.find(end_flag)
    if start_pos == -1 or end_pos == -1:
        return None, None
    if end_pos <= start_pos:
        return None, None
    command = msg[0:start_pos]
    value = msg[content_pos:end_pos]
    return command, value

def _send_by_firewater(data_list, socket):
    send_msg = ",".join(str(x) for x in data_list) + "\n"
    socket.send(send_msg.encode("utf8"))

def _send_by_justfloat(data_list, socket):
    format_string = '<' + 'f' * len(data_list)
    packed_data = struct.pack(format_string, *data_list)
    tail = b'\x00\x00\x80\x7f'
    socket.send(packed_data + tail)

def _send_thread(conn, method, socket):
    global pack
    global message
    while True:
        msg = conn.recv()
        if msg[0] == 0: # 来自track.py的消息
            message[0] = msg[1] # 循迹偏离角度
            message[1] = msg[2] # 循迹线偏离中心x坐标
            message[2] = msg[3] # 路口x坐标
            message[3] = msg[4] # 路口y坐标
            message[8] = msg[5] # 终点x坐标
            message[9] = msg[6] # 终点y坐标
        elif msg[0] == 1: # 来自detect.py的消息
            message[4] = msg[1] # 四个目标的类别ID
            message[5] = msg[2]
            message[6] = msg[3]
            message[7] = msg[4]
        pack.insert_byte(0x14)  # 包头
        for i in range(10):
            pack.insert_two_bytes(pack.num_to_bytes(message[i]))
        pack.send_packet() # 发送数据包
        try:
            if method == "firewater":
                _send_by_firewater(message, socket)
            elif method == "justfloat":
                _send_by_justfloat(message, socket)
        except:
            print("客户端断开连接")
            isConnected = False
            conn.send(isConnected)
            break

def _recv_thread(conn, socket):
    while True:
        try:
            msg = socket.recv(1024).decode("utf8")
            if len(msg) == 0:
                break
            conn.send(msg)
        except:
            break

def Listen_Thread(connect_socket):
    global server_socket
    global isConnected
    connect_socket.listen(3)
    server_socket, client_addr = connect_socket.accept()
    isConnected = True

def Empty_Thread(conn):
    global isConnected
    global message
    global pack
    while not isConnected:
        if conn.poll(0.01):
            msg = conn.recv()
            if msg[0] == 0:
                message[0] = msg[1]
                message[1] = msg[2]
                message[2] = msg[3]
                message[3] = msg[4]
                message[8] = msg[5]
                message[9] = msg[6]
            elif msg[0] == 1:
                message[4] = msg[1]
                message[5] = msg[2]
                message[6] = msg[3]
                message[7] = msg[4]
            pack.insert_byte(0x14)  # 包头
            for i in range(10):
                pack.insert_two_bytes(pack.num_to_bytes(message[i]))
            pack.send_packet() # 发送数据包


def Send_Process(conn, method="justfloat"):
    global server_socket
    global isConnected
    if method not in ("justfloat", "firewater"):
        print("发送方式不正确")
        method = "justfloat"
        print("自动更改格式为justfloat")
    isConnected = False
    connect_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connect_socket.bind(("", 11451))
    while True:    
        t0 = Thread(target=Listen_Thread, args=(connect_socket,))
        t00 = Thread(target=Empty_Thread, args=(conn,))
        t00.start()
        t0.start()
        t0.join()
        t00.join()
        isConnected = True
        print("客户端已连接")
        t1 = Thread(target=_send_thread, args=(conn, method, server_socket))
        t2 = Thread(target=_recv_thread, args=(conn, server_socket))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        isConnected = False
        try:
            server_socket.close()
        except Exception:
            pass
    
if __name__ == "__main__":
    parent_conn, child_conn = Pipe()
    p_send = Process(target=Send_Process, args=(child_conn, "justfloat"))
    p_send.start()
    while True:
        msg = parent_conn.recv()
        print("接收到数据:", msg)
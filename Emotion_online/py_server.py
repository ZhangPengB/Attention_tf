import socket
import sys
import time

def socket_server(ip='127.0.0.1', port=6666):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR, 1)
        s.bind((ip, port))
        s.listen(2)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
        
    count = 0
    print('Waiting for connection...')
    while True:
        sock, addr = s.accept()
        print('Connected!')
        # buf= sock.recv(fileinfo_size)
        
        while True:
            if count <= 5:
                sock.send(b'1')
            if count > 5:
                sock.send(b'2')
            if count > 10:
                break
            
            time.sleep(2)
            count += 1
            
        break
    sock.close()


if __name__ == '__main__':
    socket_server()


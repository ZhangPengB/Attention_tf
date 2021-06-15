import socket
import os
import sys
import struct
import tarfile


def socket_client():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('127.0.0.1', 6666))
            
        except socket.error as msg:
            print(msg)
            print(sys.exit(1))
        
        while True:
            msg = s.recv(1)
            if len(msg) != 0:
                print(msg)
            else:
                break
        break
            
            
if __name__ == '__main__':
    socket_client()

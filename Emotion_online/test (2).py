import os
from keras.utils import np_utils
# 打开exe程序
def open_exe(path):
    os.startfile(path)

# 写字符串到一个txt文件
def write_index(str_list):
    with open("/data/777.txt", "w", encoding='utf-8') as f:
        for name in str_list:
            f.write(name + '\n')
# # test
if __name__ == '__main__':
    with open("./Emotion_online/mask.txt", "r", encoding='utf-8') as f:
        mask = f.read()
    print(mask)
    if mask == '0':
        train_label = np_utils.to_categorical(0, 2)
    elif mask == '1':
        train_label = np_utils.to_categorical(1, 2)
    else:
        raise ('error: no label')

# # 多线程
# import threading
# import time
#
# def tstart(arg):
#     time.sleep(0.5)
#     print('%s running .....' %arg)
#
# if __name__ == '__main__':
#     t1 = threading.Thread(target=tstart,args=('This is thread 1',))
#     t2 = threading.Thread(target=tstart,args=('This is thread 2',))
#     t1.start()
#     t2.start()
#     print("This is main function")

# # 多进程
# from multiprocessing import Process
# import os, time
#
# def pstart(name):
#     # time.sleep(0.1)
#     print("Process name: %s, pid: %s "%(name, os.getpid()))
#
# if __name__ == "__main__":
#     subproc = Process(target=pstart, args=('subprocess',))
#     subproc.start()
#     subproc.join()
#     print("subprocess pid: %s"%subproc.pid)
#     print("current process pid: %s" % os.getpid())
from random import randint
import time

def convey_video(videos):
    file = 'videos.txt'
    with open(file, 'w') as f:
        for video in videos:
            f.write(video + '\n')
    f.close()

def read_mask():
    file = 'mask.txt'
    f = open(file, 'r')
    lines = f.read().splitlines()
    # '2'：视频播放结束
    if '2' in lines:
        f.close()
        with open(file, 'w') as f:
            mask = '2'
            f.truncate()
            f.close()
    # ‘0’：播放正向视频
    elif '0' in lines:
        mask = '0'
        f.close()
    # ‘1’：播放负向视频
    elif '1' in lines:
        mask = '1'
        f.close()
    # else: 视频暂未播放
    else:
        mask = ''
        f.close()

    return mask

def change_mask():
    file = 'mask.txt'
    while True:
        with open(file, 'w') as f:
            mask = randint(0, 5)
            f.write(str(mask) + '\n')
            f.close()
        time.sleep(1)
        if mask == 2:
            break




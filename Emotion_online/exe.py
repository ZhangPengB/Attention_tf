import os
import time
import subprocess

main = "C:\\Users\\wb\\Desktop\\matlab.exe"
subprocess.Popen(main)
time.sleep(1)
os.system('taskkill /F /IM ' + main)
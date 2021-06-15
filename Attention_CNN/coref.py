import numpy as np
from scipy.fftpack import fft, ifft
# from numpy import fft
from matplotlib import pyplot as plt

# N = 100
# x = np.arange(N)
#
# ab = np.random.randn(1, 100)
# fft_y = fft(ab)
# abs_y = np.abs(fft_y)
# angle_y = np.angle(fft_y)
#
# plt.figure(1)
# plt.plot(x, abs_y.T)
# plt.title("shuangbian  zhenfu")
#
# plt.figure(2)
# plt.plot(x, angle_y.T)
# plt.title("Shuangbian xaingwei")
# plt.show()
#
# print(fft_y.shape)
# print(fft_y[0][:10])
# print("ab=\n", ab)
# # plt.subplot()
# # plt.plot(ab.T)
# # plt.subplot(2, 1, 2)
# # plt.plot(fft_y)
# print(np.corrcoef(ab))


x = [[10, 5],
     [6, 4]]
y = [[6, 8],
     [12, 14]]
z=[[0,0],
   [1,1]]
plt.plot(x)
plt.show()
plt.plot(z,x)
plt.show()

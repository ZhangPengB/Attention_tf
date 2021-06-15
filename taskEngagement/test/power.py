from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fs = 1000
#采样点数
num_fft = 1024;

"""
生成原始信号序列

在原始信号中加上噪声
np.random.randn(t.size)
"""
t = np.arange(0, 1, 1/fs)
f0 = 100
f1 = 200
x = np.cos(2*np.pi*f0*t) + 3*np.cos(2*np.pi*f1*t) + np.random.randn(t.size)

plt.figure(figsize=(15, 12))
ax=plt.subplot(511)
ax.set_title('original signal')
plt.tight_layout()
plt.plot(x)

"""
FFT(Fast Fourier Transformation)快速傅里叶变换
"""
Y = fft(x, num_fft)
Y = np.abs(Y)

ax=plt.subplot(512)
ax.set_title('fft transform')
plt.plot(20*np.log10(Y[:num_fft//2]))

"""
功率谱 power spectrum
直接平方
"""
ps = Y**2 / num_fft
ax=plt.subplot(513)
ax.set_title('direct method')
plt.plot(20*np.log10(ps[:num_fft//2]))

"""
相关功谱率 power spectrum using correlate
间接法
"""
cor_x = np.correlate(x, x, 'same')
cor_X = fft(cor_x, num_fft)
ps_cor = np.abs(cor_X)
ps_cor = ps_cor / np.max(ps_cor)
ax=plt.subplot(514)
ax.set_title('indirect method')
plt.plot(20*np.log10(ps_cor[:num_fft//2]))
plt.tight_layout()
plt.show()

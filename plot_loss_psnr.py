import matplotlib.pyplot as plt
import numpy as np
# c = np.load('learning-rate=1e-3/loss.npz')
# d = np.load('learning-rate=1e-3/psnr.npz')
save_path = './model/ablation/'
c = np.load(save_path+'loss.npz')
d = np.load(save_path+'psnr.npz')
y = c['loss']
# 保存错数据类型了，因此转换成float
y = y.astype('float64')
x = [i for i in range(len(y))]
y1 = d['psnr']
x1 = [5*i for i in range(len(y1))]
# 画图
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.title('Loss vs. epoch')
plt.plot(x, y)
plt.figure()

plt.ylabel('psnr')
plt.xlabel('epoch')
plt.title('psnr vs. epoch')
plt.plot(x1, y1)
plt.show()
# 保存矩阵
# np.savez('loss.npz', y=y, x=x)
print(d.files)
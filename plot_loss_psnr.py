import matplotlib.pyplot as plt
import numpy as np
x = np.array([i for i in range(0,200)])
y = x*2
y1 = x**2
# 画图
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.title('Loss vs. epoch')
plt.plot(x, y)
plt.figure()

plt.ylabel('acc')
plt.xlabel('epoch')
plt.title('acc vs. epoch')
plt.plot(x, y1)
# plt.show()
# 保存矩阵
np.savez('loss.npz', y=y, x=x)
c = np.load('loss.npz')
print(c.files)
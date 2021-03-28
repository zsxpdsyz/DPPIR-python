import matplotlib.pyplot as plt
import numpy as np
import cv2

sigma = 50
dir_path = './data/CBSD68/3096.png'
# cv2的读取和显示顺序都是b，g，r
img = cv2.imread(dir_path)
d0, d1, d2 = img.shape
noise = np.random.randn(d0, d1, d2)*sigma
img_noise = noise + img
cv2.imwrite('./results/denoise/3096.png', img_noise)

dir_path = './data/Set68/test066.png'
# cv2的读取和显示顺序都是b，g，r
img = cv2.imread(dir_path, 0)
d0, d1 = img.shape
noise = np.random.randn(d0, d1)*sigma
img_noise = noise + img
cv2.imwrite('./results/denoise/test066.png', img_noise)
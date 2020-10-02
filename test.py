import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from model.IRCNN import IRCNN
import cv2
import os
# 生成去噪后的图像
def test(image, noise_image, net, DEVICE):
    net.eval()
    with torch.no_grad():
        output_x = noise_image.to(DEVICE) - net(noise_image.to(DEVICE))

    output_x = output_x.cpu().numpy()
    GT = image.cpu().numpy()
    batch, _, _, _ = GT.shape
    psnr = peak_signal_noise_ratio(output_x[0, 0, :, :], GT[0, 0, :, :], data_range=1)
    print(psnr)
    output_x = np.squeeze(output_x, axis=0)
    output_x = np.squeeze(output_x, axis=0)
    return output_x, psnr

if __name__ == '__main__':
    test_root = 'data/Set68/'
    model_path = 'model/sigma50epoch145loss220.7874.pth'
    save_img_path = 'data/sigma50/'
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
    sigma = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Use device is {DEVICE}')

    net = IRCNN(1).to(DEVICE)
    net.load_state_dict(torch.load(model_path))

    names = os.listdir(test_root)
    for name in names:
        img_path = os.path.join(test_root, name)
        image = cv2.imread(img_path, 0)
        image = (image/255).astype('float32')
        image = torch.from_numpy(image)
        # image = torch.
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 扩展2个维度
        noise = torch.randn(image.size()).mul_(sigma/255.0)
        noise_image = image + noise
        noise_img = noise_image.unsqueeze(0)
        noise_image = noise_img.unsqueeze(0)
        img = image.unsqueeze(0)
        image = img.unsqueeze(0)
        clean_img, psnr = test(image, noise_image, net, DEVICE)
        # 保存图像
        save_path = os.path.join(save_img_path, name)
        cv2.imwrite(save_path, clean_img*255)
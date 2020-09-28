# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from torch.autograd import  Variable
from model.IRCNN import IRCNN
import cv2
import numpy as np

def train(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
    img = cv2.imread('cake.jpg')
    img = np.transpose(img,(2,0,1))
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = IRCNN(3)
    net.eval
    net.to('cpu')
    img1 = torch.from_numpy(img)
    img1 = torch.unsqueeze(img1,0)
    print(img1.shape)
    output = net(img1.float())
    print(output.shape)
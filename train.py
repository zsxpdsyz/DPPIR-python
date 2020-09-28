import torch
import numpy
from utils.datasets import mydataset
from utils.datasets import TestDataset
import argparse
import os
from torch.utils.data import DataLoader
from model.IRCNN import IRCNN, Mean_Squared_Error
from torch.optim.lr_scheduler import MultiStepLR
from skimage.measure import compare_psnr
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
parse = argparse.ArgumentParser(description='IRCNN Train Parameter')
parse.add_argument('--train_data_path', default='data/train', type=str, help='path of train data')
parse.add_argument('--validation_data_path',default='data/Set68', type=str)
parse.add_argument('--batch_size', default=128, type=int)
parse.add_argument('--val_batch_size', default=10, type=int)
parse.add_argument('--epoch', default=150, type=int)
parse.add_argument('--lr', default=1e-3, type=float)
parse.add_argument('--resume', type=str)
args = parse.parse_args()
epochs = args.epoch
batch_size = args.batch_size
val_batch_size = args.val_batch_size
lr = args.lr
train_data_path = args.train_data_path
validation_data_path = args.validation_data_path

sigma = 50

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Device is {DEVICE}')
save_dir = os.path.join('model'+'_'+str(sigma))
train_dataset = mydataset(train_data_path, sigma)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataset = TestDataset(validation_data_path, sigma)
validationloader = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=False)

net = IRCNN(1)
logging.info('build net')
if args.resume:
    net.load_state_dict(torch.load(args.resume))
    logging.info(f'Resume from the model {args.resume}')
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = MultiStepLR(optimizer, [30,60,90], gamma=0.2)
criterion = Mean_Squared_Error()
def test(val_loader, net, criterion, DEVICE):
    net.eval()
    batch_loss = 0
    psnr = []
    for it, (batch_y, batch_x) in enumerate(val_loader):
        with torch.no_grad():
            output_x = batch_y - net(batch_y)
            loss = criterion(output_x, batch_x)
        # 求图像的psnr
        img_x = output_x.numpy()
        GT = batch_x.numpy()
        batch, _, _, _ = GT.shape
        for i in range(batch):
            psnr.append(compare_psnr(img_x[i,:,:,:], GT[i,:,:,:]))
        psnr_mean = numpy.mean(psnr)
        batch_loss += loss
    return batch_loss / (it + 1), psnr_mean

for epoch in range(epochs):
    scheduler.step(epoch)
    net.train()
    for batch_id, (batch_y, batch_x) in enumerate(trainloader):
        logging.info(f"Begin training, epoch = {epoch}")
        optimizer.zero_grad()
        output_x = batch_y - net(batch_y) # batch_size，channel， height，width=128x1x40x40
        loss = criterion(output_x, batch_x)
        loss.backward()
        optimizer.step()
        if batch_id % 50 == 0:
            print('Train epoch: {}\t Loss:{}'.format(epoch, loss.item()))
    if epoch % 1 == 0:
        logging.info('Begin testing')
        test_loss, psnr = test(val_loader, net, criterion, DEVICE)
        logging.info(f'Dataset mean psnr is {psnr}')
        torch.save(net.state_dict(), ('./model/'+'sigma'+str(sigma)+'epoch'+str(epoch)+'loss'+str(test_loss.item())))
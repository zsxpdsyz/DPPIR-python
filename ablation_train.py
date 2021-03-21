import torch
import numpy
from utils.datasets import mydataset
from utils.datasets import TestDataset
import argparse
import os
from torch.utils.data import DataLoader
from model.IRCNN import IRCNN, Mean_Squared_Error, AblationIRCNN
from torch.optim.lr_scheduler import MultiStepLR
from skimage.metrics import peak_signal_noise_ratio
import logging
import sys
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
parse = argparse.ArgumentParser(description='IRCNN Train Parameter')
parse.add_argument('--train_data_path', default='data/train', type=str, help='path of train data')
parse.add_argument('--validation_data_path',default='data/Set68', type=str)
parse.add_argument('--save_path',default='model', type=str)
parse.add_argument('--batch_size', default=512, type=int)
parse.add_argument('--val_batch_size', default=1, type=int)
parse.add_argument('--epoch', default=150, type=int)
parse.add_argument('--lr', default=1e-5, type=float)
parse.add_argument('--sigma', default=20, type=float, help='The level of noise')
parse.add_argument('--resume', type=str)
args = parse.parse_args()
epochs = args.epoch
batch_size = args.batch_size
val_batch_size = args.val_batch_size
save_path = args.save_path
lr = args.lr
train_data_path = args.train_data_path
validation_data_path = args.validation_data_path

sigma = args.sigma
model_save_path = save_path+'/'+str(sigma)+'/'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
total_loss = np.empty(0)
total_psnr = np.empty(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Device is {DEVICE}')
save_dir = os.path.join('model'+'_'+str(sigma))
train_dataset = mydataset(train_data_path, sigma)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataset = TestDataset(validation_data_path, sigma)
validationloader = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=False)

net = AblationIRCNN(inchannel=1).to(DEVICE)
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
            output_x = batch_y.to(DEVICE) - net(batch_y.to(DEVICE))
            loss = criterion(output_x.to(DEVICE), batch_x.to(DEVICE))
        # 求图像的psnr
        img_x = output_x.cpu().numpy()
        GT = batch_x.cpu().numpy()
        batch, _, _, _ = GT.shape
        for i in range(batch):
            psnr.append(peak_signal_noise_ratio(img_x[i,0,:,:], GT[i,0,:,:], data_range=1))
        psnr_mean = numpy.mean(psnr)
        batch_loss += loss
    return batch_loss / (it + 1), psnr_mean

# Begin training
for epoch in range(epochs):
    net.train()
    logging.info(f"Begin training, epoch = {epoch}")
    for batch_id, (batch_y, batch_x) in enumerate(trainloader):
        optimizer.zero_grad()
        output_x = net(batch_y.to(DEVICE)) # batch_size，channel， height，width=128x1x40x40
        loss = criterion(output_x.to(DEVICE), batch_x.to(DEVICE))
        loss.backward()
        optimizer.step()
        # if batch_id % 50 == 0:
    print('Train epoch: {}\t Loss:{}'.format(epoch, loss.item()))
    total_loss = numpy.append(total_loss, format(loss.item(),'0.4f'))
    np.savez(model_save_path+'loss.npz', loss=total_loss, epoch=epoch)
    scheduler.step(epoch)
    if epoch % 5 == 0:
        logging.info('Begin testing')
        test_loss, psnr = test(validationloader, net, criterion, DEVICE)
        logging.info(f'Dataset mean psnr is {psnr}')
        total_psnr = numpy.append(total_psnr, psnr)
        np.savez(model_save_path+'psnr.npz', psnr=total_psnr, epoch=epoch)
    if epoch % 20 == 0:
        torch.save(net.state_dict(), (model_save_path+'sigma'+str(sigma)+'epoch'+str(epoch)+'loss'+str(format(test_loss.item(),'0.4f'))+'.pth'))
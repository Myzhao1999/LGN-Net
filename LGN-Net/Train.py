import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from sklearn.metrics import roc_auc_score
from utils import *
import random
import argparse

parser = argparse.ArgumentParser(description="LGN-Net")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=6, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=10, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=5, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='your_dataset_directory', help='directory of data')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint_directory', help='save checkpoints')

args = parser.parse_args()



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "4"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = "4"
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = args.dataset_path+"/"+args.dataset_type+"/training/frames"


# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)


train_size = len(train_dataset)


train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)


# Model setting
from model.lgn_net import *
model = lgn(args.c, args.t_length, args.msize, args.fdim, args.mdim)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
model.cuda()

# Report the training process
log_dir = os.path.join(args.checkpoint_path, args.dataset_type)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

loss_func_mse = nn.MSELoss(reduction='none')

# Training

m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items
l1_loss, l2_loss = nn.L1Loss().cuda(), nn.MSELoss().cuda()
for epoch in range(args.epochs):

    labels_list = []
    model.train()
    start = time.time()
    for j,(imgs) in enumerate(train_batch):

        imgs = Variable(imgs).cuda() 
  
        outputs, fea, updated_fea, updated_orig, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss= model.forward(imgs[:, 0:12], m_items, True)

        optimizer.zero_grad()
        
        loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:, 12:]))
        loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
        loss.backward(retain_graph=True)

       
        optimizer.step()
    scheduler.step()
    
    print('----------------------------------------')
    print('Epoch:', epoch+1)

    print('Loss: Prediction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))

    print('Memory_items:')
    print(m_items)
    print('----------------------------------------')
    torch.save(model.state_dict(), os.path.join(log_dir, str(epoch+1)+'model.pth'))
    torch.save(m_items, os.path.join(log_dir, str(epoch+1)+'keys.pt'))
print('Training is finished')




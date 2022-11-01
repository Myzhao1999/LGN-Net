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
#import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.lgn_net import *
#from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob

import argparse
import scipy.io as scio
#Evaluate.py --method recon --t_length 1 --alpha 0.7 --th 0.015 --dataset_type ped2 --model_dir your_model.pth --m_items_dir your_m_items.pt




parser = argparse.ArgumentParser(description="LGN-Net")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')   #  loss=1, 1 :0.7->AUC:  88.10  # 0.8+th==0.01-->88.14  0.8+th==0.001-> 88.03
parser.add_argument('--gamma', type=float, default=0.009, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='your_dataset_path', help='directory of data')
parser.add_argument('--model_dir', type=str,  default='model_directory',help='directory of model')
parser.add_argument('--m_items_dir', type=str, default='memory_items_directory',help='directory of model')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = "0"
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

# Loading dataset
test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model


model = lgn(n_channel =3,  t_length = 5, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1)

model.load_state_dict(torch.load(args.model_dir))
model.cuda()
m_items = torch.load(args.m_items_dir)
labels = np.load('./LGN-Net/labels/frame_labels_'+args.dataset_type+'.npy')

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    #The first four frames are unpredictable
    labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

print(labels_list.shape)
label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()
model.eval()

for k,(imgs) in enumerate(test_batch):
    

    if k == label_length-4*(video_num+1):
        video_num += 1
        label_length += videos[videos_list[video_num].split('/')[-1]]['length']


    imgs = Variable(imgs).cuda()
    

    outputs, feas, updated_feas,updated_orig, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:12], m_items_test, False)
    mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,12:]+1)/2)).item()
    mse_feas = compactness_loss.item()

    # Calculating the threshold for updating at the test time
    point_sc = point_score(outputs, imgs[:,12:])
    


    if  point_sc < args.gamma:
        query = F.normalize(feas, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)

    psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
    feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)

# Measuring the abnormality score and the AUC
all_gt = []
anomaly_score_total_list = []


for video in sorted(videos_list):

    video_name = video.split('/')[-1]

    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                    anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

# mean psnr
a=-1
c=0
n_nor = n_abnor = 0
avg_nor = avg_abnor = 0
for video in sorted(videos_list):
    a+=1
    video_name = video.split('/')[-1]
    for  b in range(0, len(psnr_list[video_name])):
        
        if labels_list[c] == 0:
            n_nor += 1
            avg_nor += psnr_list[video_name][b]
        else:
            n_abnor += 1
            avg_abnor += psnr_list[video_name][b]
        c+=1

print("abnor_psnr==",avg_abnor/n_abnor,"nor_psnr==",avg_nor/n_nor)


anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

print('The result of ', args.dataset_type)
print('AUC: ', accuracy*100, '%')

### The complete code will be available after the paper is accepted.
### If you are interested in our work, please contact us by email to get the initial version code.

### Email:myzhao@fudan.edu.cn 
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from model. memory_module import *
from model.st_net import Predictor as pred

class Encoder(torch.nn.Module): #encoder in The Normal Prototype Branch
    def __init__(self, t_length = 5, n_channel =3):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )
        
        self.moduleConv1 = Basic(n_channel*(t_length-1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        
    def forward(self, x):

        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2) #64,64,128

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        #del tensorConv1,tensorConv2,tensorConv3,tensorPool1,tensorPool2,tensorPool3
        
        return tensorConv4

    
class Decoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):#1024--->512
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
      
        self.moduleConv = Basic(512, 512)
        self.moduleUpsample4 = Upsample(512, 256)  

        self.moduleDeconv3 = Basic(256, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(128, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128,n_channel,64)
        

        
    def forward(self, x, st_fea):
        
        tensorConv = self.moduleConv(x) #512 32 32-->512 32 32
        tensorUpsample4 = self.moduleUpsample4(tensorConv)  # 256  64  64
       
        
       
        
        tensorDeconv3 = self.moduleDeconv3( tensorUpsample4)# 256  64  64
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)# 128 128 128
        
        
       
        

        tensorDeconv2 = self.moduleDeconv2(tensorUpsample3)  # 128 128 128  
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)  # 64 256 256

        

        cat = torch.cat((tensorUpsample2, st_fea), dim = 1)
        output = self.moduleDeconv1(cat)    
        #del tensorConv, tensorUpsample4,tensorDeconv3,tensorUpsample3,cat,tensorDeconv2, tensorUpsample2 
                
        return output
    


class lgn(torch.nn.Module):
    def __init__(self, n_channel ,  t_length , memory_size, feature_dim , key_dim , temp_update , temp_gather):#n_channel =3,  t_length = 5, memory_size=10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1
        super(lgn, self).__init__()
        #print(memory_size,'memroysize')
        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        self.memory = Memory(memory_size,feature_dim, key_dim, temp_update, temp_gather)

        self.pred=pred()

        self.st_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=False),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=False),
            )


    def forward(self, x, keys,train):
        h, c ,m= self.pred( x,1) 
        h_c=torch.cat((h,m),dim=1)#spatiotemporal representations
        h_c=self.st_decoder(h_c)
        fea= self.encoder(x)

        if train:
            updated_fea, updated_orig, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_orig, h_c)
            return output, fea, updated_fea, updated_orig, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        else:
            updated_fea, updated_orig, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss = self.memory(
                fea, keys, train)
            

        
        #test
        if  train==False :
            
            output = self.decoder(updated_orig, h_c)
            
            return output, fea, updated_fea,updated_orig, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss
        
                                          


    
    

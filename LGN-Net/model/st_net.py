import model.st_lstm as convlstm

import torch
import torch.nn as nn
from torch.nn import functional as F

import copy


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(  # [2, 1, 256, 256]) # the encoder in The Spatiotemporal Branch
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=False),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=False),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=False),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=False),

            ) #64 64 128
    
        self.convlstm_num = 4
        self.convlstm_in_c = [128,128, 128, 128]
        self.convlstm_out_c =[128,128, 128, 128]
        self.convlstm_list = []
        for layer_i in range(self.convlstm_num):
            self.convlstm_list.append(convlstm.NPUnit(in_channel=self.convlstm_in_c[layer_i],
                                                      out_channels=self.convlstm_out_c[layer_i],
                                                      kernel_size=[3,3]))
        self.convlstm_list = nn.ModuleList(self.convlstm_list)



    def forward(self, x, out_len):
        batch_size = x.size()[0]
        input_len = 4




        h, c, out_pred = [], [], []
        for layer_i in range(self.convlstm_num):
            zero_state = torch.zeros(batch_size, self.convlstm_in_c[layer_i], 64, 64).to(self.device)
            h.append(zero_state)
            c.append(zero_state)
        memory = torch.zeros(batch_size, self.convlstm_in_c[layer_i], 64, 64).to(self.device)
        for seq_i in range(input_len + out_len - 1):
            if seq_i < input_len:
                input_x = x[:, seq_i*3:seq_i*3+3,  :, :]
                input_x = self.encoder(input_x)

            else:
                input_x = self.encoder(out_pred[-1])

            for layer_i in range(self.convlstm_num):
                if layer_i == 0:  # input_x=2,128,64,64
                    h[layer_i], c[layer_i], memory = self.convlstm_list[layer_i](input_x, h[layer_i], c[layer_i],memory)

                else:
                    h[layer_i], c[layer_i] ,memory= self.convlstm_list[layer_i](h[layer_i - 1], h[layer_i], c[layer_i],memory)
    
        return h[-1], c[-1],memory



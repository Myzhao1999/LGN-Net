import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
from torch.nn import functional as F

def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu
    
def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)

def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs-1):
        result = torch.cat((result, distance(a[i], b)), 0)
        
    return result

def multiply(x): #to flatten matrix into a vector 
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)

def index(batch_size, x):
    idx = torch.arange(0, batch_size).long() 
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)

def MemoryLoss(memory):

    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t))/2 + 1/2 # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)
    
    return torch.sum(sim)/(m*(m-1))


class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim,  temp_update, temp_gather):# memory_size = 10, feature_dim = 512, key_dim = 512,temp_update = 0.1, temp_gather=0.1
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        
    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem,torch.t(self.keys_var))
        similarity[:,i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)
        
        
        return self.keys_var[max_idx]
    
    def random_pick_memory(self, mem, max_indices):
        
        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices==i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)
            
        return torch.tensor(output)
    
    def get_update_query(self, mem, max_indices,                update_indices,                  score,               query,       train):
                            # (keys, gathering_indices([4096, 1]), updating_indices ([1, 10]), softmax_score_query[4096, 10], query_reshape,train)
        m, d = mem.size()  #10 512
        if train:
            query_update = torch.zeros((m,d)).cuda()    #10 512
            # random_update = torch.zeros((m,d)).cuda()
            for i in range(m):  #10
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) *query[idx].squeeze(1)), dim=0)
                else:
                    query_update[i] = 0 
        
       
            return query_update 
    
        else:
            query_update = torch.zeros((m,d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) *query[idx].squeeze(1)), dim=0)
                else:
                    query_update[i] = 0 
            
            return query_update

    def get_score(self, mem, query):#4, 32,32,512
        bs, h,w,d = query.size()
        m, d = mem.size()
        #print("m,d",m,d)  10 512
        

        score = torch.matmul(query, torch.t(mem))# b X h X w X m  #两个张量矩阵相乘，在PyTorch中可以通过torch.matmul函数实现  torch.t()是一个类似于求矩阵的转置的函数，但是它要求输入的tensor结构维度<=2D。
        #print("score",score.shape)  ([4, 32, 32, 10])
        score = score.view(bs*h*w, m)# (b X h X w) X m
        #print("score", score.shape) ([4096, 10])

        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score,dim=1)
        
        return score_query, score_memory
    
    def forward(self, query, keys, train=True): # torch.Size([4, 512, 32, 32])
        #print(self.memory_size,"memory_size")
        batch_size, dims,h,w = query.size() # b X d X h X w
        query = F.normalize(query, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d    4, 32,32,512
        
        #train
        if train:
            #losses
            separateness_loss, compactness_loss = self.gather_loss(query,keys, train)
            # read
            updated_query, updated_orig,softmax_score_query,softmax_score_memory = self.read(query, keys)
            #update
            updated_memory = self.update(query, keys, train)
            
            return updated_query,updated_orig, updated_memory, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        
        #test
        else:
            # loss
            compactness_loss, query_re, top1_keys, keys_ind = self.gather_loss(query,keys, train)
            
            # read
            updated_query, updated_orig,softmax_score_query,softmax_score_memory = self.read(query, keys)
            
            #update
            updated_memory = keys
                
               
            return updated_query, updated_orig,updated_memory, softmax_score_query, softmax_score_memory, query_re, top1_keys,keys_ind, compactness_loss
        
        
    
    def update(self, query, keys,train):
        
        batch_size, h,w,dims = query.size() # b X h X w X d
        
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        
        query_reshape = query.contiguous().view(batch_size*h*w, dims) #([4096, 512])
        
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)  ##([4096, 10])
        #print("gathering_indices",gathering_indices.shape)     ([4096, 1])
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
        #print("updating_indices ", updating_indices .shape)   ([1, 10])
        
        if train:
             
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape,train)
            updated_memory = F.normalize(query_update + keys, dim=1)
        
        else:
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)
        
        return updated_memory.detach()
        
        
    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n,dims = query_reshape.size() # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')
        
        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())
                
        return pointwise_loss
        
    def gather_loss(self,query, keys, train):
        #print("keys",keys.shape) ([10, 512])
        batch_size, h,w,dims = query.size() # b X h X w X d    [4,32, 32,512]
        if train:
            loss = torch.nn.TripletMarginLoss(margin=1.0)
            loss_mse = torch.nn.MSELoss()
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
            #([4096, 10])
            query_reshape = query.contiguous().view(batch_size*h*w, dims)  #4096 512
        
            _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)
            #print("gather",gathering_indices.shape) [4096, 2]
        
            #1st, 2nd closest memories
            pos = keys[gathering_indices[:,0]]
            #print("pos",pos.shape) [4096, 512]

            neg = keys[gathering_indices[:,1]]
            top1_loss = loss_mse(query_reshape, pos.detach())
            gathering_loss = loss(query_reshape,pos.detach(), neg.detach())
            
            return gathering_loss, top1_loss
        
            
        else:
            loss_mse = torch.nn.MSELoss()
        
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        
            query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
            _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        
            gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())
            
            return gathering_loss, query_reshape, keys[gathering_indices].squeeze(1).detach(), gathering_indices[:,0]
            
        
        
    
    def read(self, query, updated_memory):  #4, 32,32,512
        batch_size, h,w,dims = query.size() # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)
        #torch.Size([4096, 10]) torch.Size([4096, 10])
        query_reshape = query.contiguous().view(batch_size*h*w, dims) #4096,512

        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory) # (b X h X w) X d   ([4096, 10]) ([10,512])
        updated_orig=concat_memory
        updated_query = torch.cat((query_reshape, concat_memory), dim = 1) # (b X h X w) X 2d   #4096,512
        updated_query = updated_query.view(batch_size, h, w, 2*dims)  #4, 32,32,1024
        updated_query = updated_query.permute(0,3,1,2)#4, 1024 ,32,32,

        updated_orig = updated_orig.view(batch_size, h, w,dims)  # 4, 32,32,1024
        updated_orig = updated_orig.permute(0, 3, 1, 2)  # 4, 1024 ,32,32,
        
        return updated_query,updated_orig ,softmax_score_query, softmax_score_memory
    
    

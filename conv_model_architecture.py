import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np

torch.manual_seed(43)
torch.cuda.manual_seed(542)
torch.cuda.manual_seed_all(117)

class Conv_block(nn.Module):
    
    def init_weights(self):
        
        print('init weights')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            
    def __init__(self, in_ch, out_ch, filter_size):
        super(Conv_block, self).__init__()
        
        self.convblock = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_ch, filter_size, kernel_size=3, stride=1, padding=0)),
                ('bn1', nn.BatchNorm2d(filter_size)),
                ('relu1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
                ('conv2', nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=0)),
                ('bn2', nn.BatchNorm2d(filter_size)),
                ('relu2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
                ('conv3', nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=0)),
                ('bn3', nn.BatchNorm2d(filter_size)),
                ('relu3', nn.ReLU()),
                ('pool3', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
                ('conv4', nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=0)),
                ('bn4', nn.BatchNorm2d(filter_size)),
                ('relu4', nn.ReLU()),
                ('pool4', nn.MaxPool2d(kernel_size=2, stride=1, padding=0))
        ]))
        self.add_module('fc', nn.Linear(filter_size * 5 * 5, out_ch))
        self.init_weights()
        
        
    def conv(self, input, weight, bias, stride=1, padding=0):        
        conv = F.conv2d(input, weight, bias, stride=1, padding=0)
        return conv
        
    def batchnorm(self, input, weight, bias):
        running_mean = torch.zeros(np.prod(np.array(input.data.size()[1]))).cuda()
        running_var = torch.ones(np.prod(np.array(input.data.size()[1]))).cuda()
        batchnorm = F.batch_norm(input, running_mean, running_var, weight, bias, training=True)
        return batchnorm
    
    def relu(self, input):
        relu = F.relu(input)
        return relu
        
    def maxpool(self, input, kernel_size=2, stride=2, padding=0 ):
        maxpool = F.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding)        
        return maxpool
    
    def forward(self, x, weights = None):
        if weights == None:
            x = self.convblock(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            weight_idx = list(weights.keys())   
            x = self.conv(x, weights[weight_idx[0]], weights[weight_idx[1]])
            x = self.batchnorm(x, weights[weight_idx[2]], weights[weight_idx[3]])
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv(x, weights[weight_idx[4]],  weights[weight_idx[5]])
            x = self.batchnorm(x,  weights[weight_idx[6]],  weights[weight_idx[7]])
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv(x, weights[weight_idx[8]],  weights[weight_idx[9]])
            x = self.batchnorm(x, weights[weight_idx[10]],  weights[weight_idx[11]])
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv(x, weights[weight_idx[12]],  weights[weight_idx[13]])
            x = self.batchnorm(x, weights[weight_idx[14]],  weights[weight_idx[15]])
            x = self.relu(x)
            x = self.maxpool(x, kernel_size=2, stride=1)
            x = x.view(x.size(0), -1)
            x = F.linear(x, weights[weight_idx[16]],  weights[weight_idx[17]])

        return x

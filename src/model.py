from turtle import forward
import torch
from typing import Tuple
from torch import nn
from torch.nn import functional as F

def get_model(model_name):

    name_to_model = {
        "LeNet": LeNet,
        "CNNCIFAR":CNNCIFAR,
        "TextCNN": TEXTCNN,
    }

    if model_name in name_to_model.keys():
        return name_to_model[model_name]
    else:
        print('ERROR! Model name in {}.'.format(name_to_model.keys()))
        exit()

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # self.fc = nn.Sequential(
        #     nn.Linear(16 * 4 * 4, 120),
        #     nn.ReLU(),
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, 10)
        # )

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.fc(x)
        return x

class CNNCIFAR(nn.Module):
    def __init__(self):
        
        super(CNNCIFAR,self).__init__()       # RGB 3*32*32
        self.conv1 = nn.Conv2d( 3, 15,3)     # 输入3通道，输出15通道，卷积核为3*3
        self.conv2 = nn.Conv2d(15, 75,4)    # 输入15通道，输出75通道，卷积核为4*4
        self.conv3 = nn.Conv2d(75,375,3)    # 输入75通道，输出375通道，卷积核为3*3
        self.fc1 = nn.Linear(1500,400)       # 输入2000，输出400
        self.fc2 = nn.Linear(400,120)        # 输入400，输出120
        self.fc3 = nn.Linear(120, 84)        # 输入120，输出84
        self.fc4 = nn.Linear(84, 10)         # 输入 84，输出 10（分10类）
        # self.fc = nn.Sequential(
        #     nn.Linear(375*2*2, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, 120),
        #     nn.ReLU(),
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, 10)
        # )
 
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)      # 3*32*32  -> 150*30*30  -> 15*15*15
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)      # 15*15*15 -> 75*12*12  -> 75*6*6
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)      # 75*6*6   -> 375*4*4   -> 375*2*2
        x = x.view(x.size()[0],-1)                      # 将375*2*2的tensor打平成1维，1500
        x = F.relu(self.fc1(x))                         # 全连接层 1500 -> 400
        x = F.relu(self.fc2(x))                         # 全连接层 400 -> 120
        x = F.relu(self.fc3(x))                         # 全连接层 120 -> 84
        x = self.fc4(x)                                 # 全连接层 84  -> 10
        # x = self.fc(x)
        return x

class TEXTCNN(nn.Module):
    
    def __init__(self, vocab_size, output_dim, pad_idx, embedding_dim=300, n_filters=100, filter_sizes=[2,3,4], 
                 dropout=0.1):
        super(TEXTCNN, self).__init__()

        self.embed=nn.Embedding(vocab_size,embedding_dim,pad_idx)

        self.convs=nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,out_channels=n_filters
                                            ,kernel_size=fs) for fs in filter_sizes])

        self.fc=nn.Linear(len(filter_sizes)*n_filters,output_dim)
    
    def forward(self,text):
        #text:(batch,seq)
        embeded=self.embed(text)
        #embeded:(batch,seq,embedding)
        
        embeded=embeded.permute(0,2,1)
        # embeded:(batch,embedding,seq)
        #print(embeded.shape)
        conveds=[torch.relu(conv(embeded)) for conv in self.convs]
        #conved:(batch,n_filters,seq-kernel_size+1)
        #print([ot.shape for ot in conveds])

        pooled=[F.max_pool1d(conved,conved.shape[-1]).squeeze(-1) for conved in conveds]
        #print([ot.shape for ot in pooled])
        #pooled:(batch,n_filters)
        #print(pooled.shape)
        cat=torch.cat(pooled,dim=1)
        #cat:(batch,n_filters*len(filter_size))
        #print(cat.shape)
        return self.fc(cat)
        #（batch,output_dim）

import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear, Sequential
from torch_geometric.nn.conv import EGConv,GATConv
from torch.nn import Linear, Dropout,ELU
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, num_heads, num_bases):
        super().__init__()
        
        aggregators = ['mean','sum','std','max','min',]
        
        self.Conv1D = nn.Sequential(
                nn.Conv1d(hidden_channels,256,5,dilation = 4,padding = 'same'),
                nn.BatchNorm1d(256)
        ) 

        self.Conv1D_2 = nn.Sequential(
                nn.Conv1d(256,256,5,dilation = 4,padding = 'same'),
                nn.BatchNorm1d(256)
        ) 

        self.Conv1D_dilated = nn.Sequential(
                nn.Conv1d(256,512,3,dilation = 4,padding = 'same'),
                nn.BatchNorm1d(512)
        )
        self.dropout = Dropout(0.25)

        hidden_channels = 512
        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.attention = GATConv(in_channels=512,out_channels=256,heads=2)
        self.attention_batch = nn.BatchNorm1d(512)

        for _ in range(num_layers):
            self.convs.append(
                EGConv(hidden_channels, hidden_channels, aggregators,
                       num_heads, num_bases))
            self.norms.append(BatchNorm1d(hidden_channels))
    
        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2, bias=False),
            BatchNorm1d(hidden_channels // 2),
            ELU(inplace=True),
            Linear(hidden_channels // 2, hidden_channels // 4, bias=False),
            BatchNorm1d(hidden_channels // 4),
            ELU(inplace=True),
            Linear(hidden_channels // 4, hidden_channels // 8, bias=False),
            BatchNorm1d(hidden_channels // 8),
            ELU(inplace=True),
            Linear(hidden_channels // 8, 1),
        )

    def forward(self, data):
        x, edge_index, batch  = data.x.cuda(), data.edge_index.cuda(), data.batch
        batch_size = batch.max().item()+1

        x = x.view(-1,140,512).permute(0,2,1)
        x1 = F.relu(self.Conv1D(x))
        x1 = F.relu(self.Conv1D_2(x1))
        x1 = F.relu(self.Conv1D_dilated(x1))
        x  = x1 + x
        x = x.permute(0,2,1).reshape(140*batch_size,self.hidden_channels)
    
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = h.relu_()
            x = x + h
        x1,attention = self.attention(x,edge_index,return_attention_weights=True)
        x = x + self.attention_batch(x1.relu_())
        
        return F.sigmoid(self.mlp(x))



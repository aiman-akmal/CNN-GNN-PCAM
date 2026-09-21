import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TopKPooling, global_mean_pool

class HybridGraphSAGE(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, out_channels=2):
        super(HybridGraphSAGE, self).__init__()
        
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        
        self.topk_pool = TopKPooling(hidden_channels, ratio=0.5)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.sage1(x, edge_index))
        x = F.relu(self.sage2(x, edge_index))
        
        x, edge_index, _, batch, _, _ = self.topk_pool(x, edge_index, None, batch)
        
        x = global_mean_pool(x, batch) 
        
        out = self.classifier(x)
        return out
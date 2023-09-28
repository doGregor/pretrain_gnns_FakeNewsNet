import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv
from torch_geometric.nn import global_mean_pool


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, num_layers=2, num_attention_heads=1, out_cat=False, dropout=False):
        super(HGT, self).__init__()
        # torch.manual_seed(12345)

        self.metadata = metadata
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(-1, hidden_channels, metadata, num_attention_heads, group='sum')
            self.convs.append(conv)
            
        self.out_cat = out_cat
        if out_cat:
            self.lin_out = Linear(-1, hidden_channels)
        
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        if self.out_cat:
            x_dict = {key: global_mean_pool(x, batch_dict[key]) for key, x in x_dict.items()}
            x_dict = torch.cat([x_dict[node_type] for node_type in self.metadata[0]], dim=1)
            x_dict = self.lin_out(x_dict).relu_()

        return x_dict

    
class NodeDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, node_type='tweet'):
        super(NodeDecoder, self).__init__()
        self.node_type = node_type
        if node_type is None:
            self.lin_dict = torch.nn.ModuleDict()
            self.lin_dict['tweet'] = torch.nn.Linear(in_channels, out_channels)
            self.lin_dict['user'] = torch.nn.Linear(in_channels, out_channels)
        else:
            self.lin = torch.nn.Linear(in_channels, out_channels)
        
    def forward(self, x_dict):
        if self.node_type is None:
            for node_type, x in x_dict.items():
                if node_type in ['tweet', 'user']:
                    # x = F.dropout(x, p=0.5, training=self.training)
                    x_dict[node_type] = self.lin_dict[node_type](x)
            x = x_dict
        else:
            x = x_dict[self.node_type]
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)
        return x
    

class GraphDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphDecoder, self).__init__()
        self.lin = torch.nn.Linear(in_channels * 3, out_channels)
        
    def forward(self, x_dict, batch_dict):
        x_dict = {key: global_mean_pool(x, batch_dict[key]) for key, x in x_dict.items()}
        x = torch.cat([x_dict['article'], x_dict['tweet'], x_dict['user']], dim=1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)
        return x

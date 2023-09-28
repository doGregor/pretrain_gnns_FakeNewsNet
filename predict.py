import numpy as np
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear, HGTConv
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, num_layers=2, num_attention_heads=1, out_cat=False):
        super(HGT, self).__init__()
        # torch.manual_seed(12345)

        self.metadata = metadata
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_attention_heads, group='sum')
            self.convs.append(conv)
            
        self.out_cat = out_cat
        if out_cat:
            self.lin_out = Linear(-1, hidden_channels)
        
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        if self.out_cat:
            x_dict = {key: global_mean_pool(x, batch_dict[key]) for key, x in x_dict.items()}
            x_dict = torch.cat([x_dict[node_type] for node_type in self.metadata[0]], dim=1)
            x_dict = self.lin_out(x_dict).relu_()

        return x_dict


def eval_model(model, test_loader, print_classification_report=False):
    model.eval()
    correct = 0
    true_y = []
    pred_y = []
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        data.to(DEVICE)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred_y.append(pred.cpu().detach().numpy())
        correct += int((pred == data['article'].y).sum())  # Check against ground-truth labels.
        true_y.append(data['article'].y.cpu().detach().numpy())
    if print_classification_report:
        print(classification_report(np.concatenate(true_y), np.concatenate(pred_y), digits=5))
    return accuracy_score(np.concatenate(true_y), np.concatenate(pred_y)), precision_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro'), recall_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro'), f1_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro')


def zero_shot_pred(pretrained_name, data_list, folder_name='models_separate', num_hidden=64, manual_seed=True,
                   random_seed=42):
    data_loader = DataLoader(data_list, batch_size=32, shuffle=False)

    model_encoder = HGT(hidden_channels=num_hidden, metadata=data_list[0].metadata())
    if pretrained_name is not None:
        model_path = folder_name + '/' + pretrained_name + '.pth'
        state = torch.load(model_path)
        model_encoder.load_state_dict(state)

    class PretrainedModel(torch.nn.Module):
        def __init__(self, encoder, hidden_channels=num_hidden, out_channels=2):
            super(PretrainedModel, self).__init__()
            if manual_seed:
                torch.manual_seed(random_seed)
            
            self.encoder = encoder
            self.decoder = torch.nn.Linear(hidden_channels * 3, out_channels)

        def forward(self, x_dict, edge_index_dict, batch_dict):
            x_dict = self.encoder(x_dict, edge_index_dict, batch_dict)
            x_dict = {key: global_mean_pool(x, batch_dict[key]) for key, x in x_dict.items()}
            x = torch.cat([x_dict['article'], x_dict['tweet'], x_dict['user']], dim=1)
            # x = F.dropout(x, p=0.2, training=self.training)
            x = self.decoder(x)
            return x

    new_model = PretrainedModel(model_encoder)
    new_model.to(DEVICE)
    
    acc, p, r, f1 = eval_model(new_model, data_loader, print_classification_report=True)
    return acc, p, r, f1

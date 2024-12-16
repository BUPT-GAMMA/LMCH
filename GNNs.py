import torch.nn as nn
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self.n_layers = model_config['gnn_n_layers']
        self.convs = nn.ModuleList([])
        # self.linear_in = nn.Linear(model_config['lm_input_dim'], self.hidden_dim)
        self.linear_in = nn.Linear(self.hidden_dim, self.hidden_dim)
  
        for i in range(self.n_layers):
            self.convs.append(RGCNConv(self.hidden_dim, self.hidden_dim, model_config['n_relations']))

        self.dropout = nn.Dropout(model_config['dropout'])
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, model_config['out_channels'])
        

    def forward(self, x, edge_index, edge_type):
        x = self.linear_in(x)
        x = self.dropout(x)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.activation(x)
        x = self.linear_pool(x)

        y = self.activation(x)
        z = self.dropout(y)

        return x, self.linear_out(z)

        # return self.linear_out(z)
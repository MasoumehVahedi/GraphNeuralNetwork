import torch
from torch import nn
from torch_geometric.nn import GCNConv, knn_graph
import torch.nn.parallel
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv



class GCN(torch.nn.Module):
    """Graph Convolutional Network"""

    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1, k=20, return_embeds=False):
        super(GCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.return_embeds = return_embeds # Skip classification layer and return node embeddings


    def reset_parameters(self):
        # If you do not call you layers gcn1, gcn2, and gcn3, then please
        # change the names of the layers in the following lines accordingly
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()

    def forward(self, x, c, adj_t, edge_w):
        x = x.float()
        c = c.float()
        if torch.is_tensor(adj_t) & torch.is_tensor(edge_w):
            edge_index = adj_t
            edge_weight = edge_w
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            edge_weight = edge_graph_weight(c, edge_index).to(self.device)

        h1 = self.gcn1(x, edge_index, edge_weight)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, training=self.training)

        h2 = self.gcn2(h1, edge_index, edge_weight)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, training=self.training)

        # If return_embeds is True, then skip the last softmax layer
        output = x if self.return_embeds else self.fc(h2)

        return output


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1, k=20):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.gat1 = GATv2Conv(input_dim, hidden_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, c, adj_t, edge_w):
        x = x.float()
        c = c.float()
        if torch.is_tensor(adj_t) & torch.is_tensor(edge_w):
            edge_index = adj_t
            edge_weight = edge_w
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            edge_weight = edge_graph_weight(c, edge_index).to(self.device)

        h1 = self.gat1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.6, training=self.training)
        h2 = self.gat2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.6, training=self.training)

        out = self.fc(h2)

        return out


class LossWrapper(nn.Module):
    def __init__(self, model, task_num=2, loss='mse', uw=True, lamb=0.5, k=20, batch_size=500):
        super(LossWrapper, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.task_num = task_num
        self.uw = uw
        self.lamb = lamb
        self.k = k
        self.batch_size = batch_size
        if task_num > 1:
          self.log_vars = nn.Parameter(torch.zeros((task_num)))
        if loss=="mse":
          self.criterion = nn.MSELoss()
        elif loss=="l1":
          self.criterion = nn.L1Loss()

    def forward(self, input, targets, coords, edge_index, edge_weight, morans_input):

        if self.task_num==1:
          outputs = self.model(input, coords, edge_index, edge_weight)
          loss = self.criterion(targets.float().reshape(-1),outputs.float().reshape(-1))
          return loss

        else:
          outputs1, outputs2 = self.model(input, coords, edge_index, edge_weight)
          if self.uw:
            precision1 = 0.5 * torch.exp(-self.log_vars[0])
            loss1 = self.criterion(targets.float().reshape(-1),outputs1.float().reshape(-1))
            loss1 = torch.sum(precision1 * loss1 + self.log_vars[0], -1)

            precision2 = 0.5 * torch.exp(-self.log_vars[1])
            loss2 = self.criterion(targets2.float().reshape(-1),outputs2.float().reshape(-1))
            loss2 = torch.sum(precision2 * loss2 + self.log_vars[1], -1)

            loss = loss1 + loss2
            loss = torch.mean(loss)
            return loss, self.log_vars.data.tolist()
          else:
            loss1 = self.criterion(targets.float().reshape(-1),outputs1.float().reshape(-1))
            loss2 = self.criterion(targets2.float().reshape(-1),outputs2.float().reshape(-1))
            loss = loss1 + self.lamb * loss2
            return loss
import torch
from torch import nn
from torch_cluster import knn_graph
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from Data_utils import get_data, edge_graph_weight, KNN_to_adj_matrix
from Data_utils import MyDataset
from gnn_model import GAT, GCN, LossWrapper



def train(path):
    # Set random seed
    np.random.seed(42)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_size = 0.7
    k = 20

    c, x, y = get_data(path)
    x = torch.ones(y.shape[0], 1)
    n = x.shape[0]
    n_train = np.round(n * train_size).astype(int)
    n_test = (n - n_train).astype(int)
    indices = np.arange(n)
    _, _, _, _, idx_train, idx_test = train_test_split(x, y, indices, test_size=(1 - train_size), random_state=42)

    train_x, test_x = x[idx_train], x[idx_test]
    train_y, test_y = y[idx_train], y[idx_test]
    train_c, test_c = c[idx_train], c[idx_test]
    train_dataset, test_dataset = MyDataset(train_x, train_y, train_c), MyDataset(test_x, test_y, test_c)

    batch_size = len(idx_train)
    train_edge_index = knn_graph(train_c, k=k).to(device)

    train_edge_weight = edge_graph_weight(train_c, train_edge_index).to(device)
    test_edge_index = knn_graph(test_c, k=k).to(device)
    test_edge_weight = edge_graph_weight(test_c, test_edge_index).to(device)
    train_moran_weight_matrix = KNN_to_adj_matrix(train_edge_index, batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # model = GCN(input_dim=train_x.shape[1], k=20, return_embeds=False).to(device)
    model = GAT(input_dim=train_x.shape[1], k=20).to(device)
    model = model.float()

    MAT = True
    if MAT:
        task_num = 2
    else:
        task_num = 1

    train_crit = "mse"
    n_epochs = 100
    lamb = 0.5
    batch_size = 1280
    lr = 0.0002

    loss_wrapper = LossWrapper(model, task_num=1, loss=train_crit, uw="True", lamb=lamb, k=k, batch_size=batch_size).to(device)
    optimizer = torch.optim.Adam(loss_wrapper.parameters(), lr=lr)
    score1 = nn.MSELoss()
    score2 = nn.L1Loss()


    writer = SummaryWriter('runs/GNN_experiment')

    # Training loop
    it_counts = 0
    for epoch in range(1, n_epochs + 1):
        for batch in train_loader:
            model.train()
            it_counts += 1
            x = batch[0].to(device).float()
            y = batch[1].to(device).float()
            c = batch[2].to(device).float()

            optimizer.zero_grad()

            loss = loss_wrapper(x, y, c, train_edge_index, train_edge_weight, train_y)
            loss.backward()
            optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(test_dataset.features).to(device),
                             torch.tensor(test_dataset.coords).to(device), test_edge_index, test_edge_weight)
            test_score1 = score1(torch.tensor(test_dataset.target).reshape(-1).to(device), pred.reshape(-1))
            test_score2 = score2(torch.tensor(test_dataset.target).reshape(-1).to(device), pred.reshape(-1))

            print("Epoch [%d/%d] - Loss: %f - Test score (MSE): %f - Test score (MAE): %f" % (
            epoch, n_epochs, loss.item(), test_score1.item(), test_score2.item()))

            writer.add_scalar('loss/train', loss.item(), it_counts)
            # writer.add_scalar('accuracy/train', train_accuracy.item(), it_counts)
            writer.add_scalar('Test score (MSE)', test_score1.item(), it_counts)
            writer.add_scalar('Test score (MAE)', test_score2.item(), it_counts)



if __name__ == "__main__":
    path = "/content/sample_data/3D_spatial_network.csv"
    train(path)
    # % load_ext tensorboard
    # % tensorboard - -logdir = runs
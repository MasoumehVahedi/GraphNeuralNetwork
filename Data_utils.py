import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from scipy.stats import wasserstein_distance
from math import radians, cos, sin, asin, sqrt

import torch
from torch import nn
from torch_geometric.nn import GCNConv, knn_graph
from torch.utils.data import Dataset, DataLoader



def normalize_vector(v, min_value=0):
  v_min = np.min(v)
  v_max = np.max(v)

  if v_min == 0 and v_max == 0:
    return v
  elif min_value == -1:
    v_normalize = 2 * ((v - v_min)/(v_max - v_min)) - 1
  elif min_value == 0:
    v_normalize = ((v - v_min)/(v_max - v_min))
  return v_normalize


def get_data(path, norm_min_val=0):
  # In this dataset, features at location is empty (None)
  df = pd.read_csv(path)
  data = df[:10000]
  data.columns = ["id","x","y","z"]
  # to get coordinates
  coords = np.array(data[["x","y"]])
  y = np.array(data[["z"]])
  # Normalize output
  y = normalize_vector(y, norm_min_val)
  return torch.tensor(coords), None, torch.tensor(y)

def degree2radians(x):
  return x * math.pi / 180



def pool2cart(lat, long):
  x = np.cos(lat) * np.cos(long)
  y = np.cos(lat) * np.sin(long)
  z = np.sin(lat)
  cart_coord = np.column_stack((x, y, z))
  return cart_coord


def cal_haversine_dist(long1, lat1, long2, lat2):
  # Calculate the great circle distance between two points using haversine distance
  coordinates = [long1, lat1, long2, lat2]
  long1, lat1, long2, lat2 = map(radians, coordinates)
  # haversine
  dist_long = long2 - long1
  dist_lat = lat2 - lat1
  a = sin(dist_lat/2)**2 + cos(lat1) * cos(lat2) * sin(dist_long/2)**2
  c = 2 * asin(sqrt(a))
  r = 6371
  return c * r * 1000


def cal_euclidean_dist(x1, y1, x2, y2):
  return math.sqrt(((x1-x2)**2)+((y1-y2)**2))


def cal_dist_3d_point(x1, y1, z1, x2, y2, z2):
  return math.sqrt(math.pow(x2 - x1, 2) +
                  math.pow(y2 - y1, 2) +
                  math.pow(z2 - z1, 2)* 1.0)


def new_dist(point1, point2, num_dim_dist="great_circle"):
  # Distance options are ["great_circle" (2D only), "euclidean", "wasserstein" (for higher-dimensional coordinate embeddings)]
  if point1.shape[0]==2:
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    if num_dim_dist=="euclidean":
      dist = cal_euclidean_dist(x1, y1, x2, y2)
    else:
      dist = cal_haversine_dist(x1,y1,x2,y2)

  elif point1.shape[0]==3:
    x1, y1, z1 = point1[0], point1[1], point1[2]
    x2, y2, z2 = point2[0], point2[1], point2[2]
    dist = cal_dist_3d_point(x1, y1, z1, x2, y2, z2)

  elif point1.shape[0]>3:
    if num_dim_dist=="wasserstein":
      dist = wasserstein_distance(point1.reshape(-1).detach(), point2.reshape(-1).detach())
    else:
      dist = torch.pow(point1.reshape(1,1,-1) - point2.reshape(1,1,-1), 2).sum(2)
  return dist


def edge_graph_weight(x, edge_index):
    """Graph weight for each edge"""
    node_to = edge_index[0]
    node_from = edge_index[1]
    edge_weight = []
    for i in range(len(node_to)):
      edge_weight.append(new_dist(x[node_to[i]], x[node_from[i]]))
    max_value = max(edge_weight)
    rng = max_value - min(edge_weight)
    edge_weight = [(max_value - elem) / rng for elem in edge_weight]
    return torch.Tensor(edge_weight)


def KNN_to_adj_matrix(knnGraph, num):
    """KNN graph to adjacency matrix"""
    adj_matrix = torch.zeros(num, num, dtype=float)
    for i in range(len(knnGraph[0])):
      node_to = knnGraph[0][i]
      node_from = knnGraph[1][i]
      adj_matrix[node_to, node_from] = 1
    return adj_matrix.T


def normal_torch(tensor, min_val=0):
    t_min = torch.min(tensor)
    t_max = torch.max(tensor)
    if t_min == 0 and t_max == 0:
      return torch.tensor(tensor)
    if min_val == -1:
      tensor_norm = 2 * ((tensor - t_min) / (t_max - t_min)) - 1
    if min_val== 0:
      tensor_norm = ((tensor - t_min) / (t_max - t_min))
    return torch.tensor(tensor_norm)


class MyDataset(Dataset):
    def __init__(self, x, y, coords):
      self.features = x
      self.target = y
      self.coords = coords

    def __len__(self):
      return len(self.features)

    def __getitem__(self, idx):
      return torch.tensor(self.features[idx]), torch.tensor(self.target[idx]), torch.tensor(self.coords[idx])

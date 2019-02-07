from torch import nn
from torch.nn import functional as F
from tochvision import models 

# num_points - number of points put into the network 
# K - number of inputs (K should be 3 for XYZ coordinates, but can be larger if normals, colors, etc are included)
def Seg_PointNet(num_points=2000, K=3):
    layers = []
    # Multilayer perceptrons with shared weights are implemented as 
	# convolutions. This is because we are mapping from K inputs to 64 
	# outputs, so we can just consider each of the 64 K-dim filters as 
	# describing the weight matrix for each point dimension (X,Y,Z,...) to
	# each index of the 64 dimension embeddings
    layers.append(nn.Sequential(
			nn.Conv1d(K, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU())) # mlp1

	layers.append(nn.Sequential(
			nn.Conv1d(64, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Conv1d(128, 1024, 1),
			nn.BatchNorm1d(1024),
			nn.ReLU())) # mlp2

    layers.append(nn.MaxPool1d(num_points))

    layers.append(nn.Sequential(
            nn.Conv1d(N, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 2, 1)
            nn.BatchNorm1d(2),
            nn.ReLU()
        )) # mlp3

    return layers

# N - number of points put into the network
# k - onehot vector length
def T_Net(N, k):
    
    layers = []
    
    layers.append(nn.Sequential(
			nn.Conv1d(3, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Conv1d(128, 256, 1),
			nn.BatchNorm1d(256),
			nn.ReLU(),
            nn.Conv1d(256, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU()
            )) # mlp1

    layers.append(nn.MaxPool1d(N))

    layers.append(nn.Sequential(
            nn.Linear(512 + k, 256),
            nn.Linear(256, 128),
            mm.Linear(128, 3)
        )) # fc 
    return layers

# M - Number of points put into the network
# k - onehot vector size
def Box_Estimation_Net(M, k, NS, NH):
    layers = [] 

    layers.append(nn.Sequential(
            nn.Conv1d(3, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU(),
            nn.Conv1d(128, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU(),
            nn.Conv1d(128, 256, 1),
			nn.BatchNorm1d(256),
			nn.ReLU(),
            nn.Conv1d(256, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU()
        )) # mlp1

    layers.append(nn.MaxPool1d(M))

    layers.append(nn.Sequential(
            nn.Linear(512 + k, 256)
            nn.Linear(256, 3 + 4*NS + 2*NH)
        )) # fc

    return layers
        
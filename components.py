from torch import nn
from torch.nn import functional as F
from tochvision import models 

class Seg_PointNet(nn.Module):
    # k - number of classes
    def __init__(self, num_points=2000, K=3):
        super(Seg_PointNet, self).__init__()

        # Multilayer perceptrons with shared weights are implemented as 
		# convolutions. This is because we are mapping from K inputs to 64 
		# outputs, so we can just consider each of the 64 K-dim filters as 
		# describing the weight matrix for each point dimension (X,Y,Z,...) to
		# each index of the 64 dimension embeddings
        self.mlp1 = nn.Sequential(
			nn.Conv1d(K, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU())

		self.mlp2 = nn.Sequential(
			nn.Conv1d(64, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Conv1d(128, 1024, 1),
			nn.BatchNorm1d(1024),
			nn.ReLU())

        self.mlp3 = nn.Sequential(
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
            nn.Conv1d(128, 2, 1)
            nn.BatchNorm1d(2),
            nn.ReLU()
        )


    def forward(self, x):

        # Number of points put into the network
		N = x.shape[2]

        # Output is 64 x N
        x1 = self.mlp1(x)

        # Output is 1024 x N
        x2 = self.mlp2(x1)

        # Output should be B x (512 + k) x 1 --> B x (512 + k) (after squeeze)
		global_feature = F.max_pool1d(x2, N).squeeze(2)

        # Concatenating global feature vector with length 1024,
        # x1 with shape 64 x N and one-hot vector with length k
        # Output: N x (1088 + k)
        features = torch.cat([one_hot, x2, global_feature], 1)

        # Output is 2 x N
        output_scores = self.mlp3(features)

        # Output is 1 x N
        object_prob = nn.Conv1d(2, 1, 1)(output_scores)

        return object_prob


class T_Net(nn.Module):
    def __init__(self, k):
        
        self.k = k
        
        self.mlp1 = nn.Sequential(
			nn.Conv1d(3, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Conv1d(128, 256, 1),
			nn.BatchNorm1d(256),
			nn.ReLU(),
            nn.Conv1d(256, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Linear(512 + k, 256),
            nn.Linear(256, 128),
            mm.Linear(128, 3)
        )

        
    def forward(self, x):

		batchsize = x.size()[0]
        # Number of points put into the network
        M = x.size()[2]
        # Output is 512 x M
        x = self.mlp1(x)
        # Output should be B x (512 + k) x 1 --> B x (512 + k) (after squeeze)
		global_feature = F.max_pool1d(x, M).squeeze(2)
        # Output 
        residual_center = self.fc(global_feature)

        return residual_center


def Box_Estimation_Net(nn.Module):
    def __init__(self, k):

        self.k = k 

        self.mlp = nn.Sequential(
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
        )

        self.fc = nn.Sequential(
            nn.Linear(512 + self.k, 256)
            nn.Linear(256, 3 + 4*self.NS + 2*self.NH)
        )
    
    
    def forward(self, x):
        
        batchsize = x.size()[0]
        # Number of points put into the network
        M = x.shape[2]
        
        # Output is 512 x M
        x = self.mlp(x)

        # Output should be B x 512 x 1 --> B x (512 + k) (after concatenation and squeeze)
        global_features = torch.cat([F.max_pool1d(x, M).squeeze(2), self.k], 1)

        box_parameters = self.fc(global_features)

        return box_parameters

        
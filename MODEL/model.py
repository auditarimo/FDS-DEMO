import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class FraudDetectionGNN(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, heads=4):
        super(FraudDetectionGNN, self).__init__()

        # Graph Attention Layers
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=heads, edge_dim=num_edge_features)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=num_edge_features)

        # Edge classifier (fraud detection)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        # Node feature propagation
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        # Edge classification (fraud prediction)
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)
        return self.edge_classifier(edge_features)
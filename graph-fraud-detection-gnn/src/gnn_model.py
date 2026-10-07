import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal Graph Convolutional Layer simulation
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_matrix):
        # Aggregate neighbor features via adjacency matrix
        support = self.linear(x)
        output = torch.matmul(adj_matrix, support)
        return F.relu(output)

class FraudGNN(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, num_classes=2):
        super().__init__()
        self.gcn1 = GraphConvLayer(input_dim, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, num_classes)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    model = FraudGNN()
    # 5 nodes with 16 features each
    nodes = torch.randn(5, 16)
    adj = torch.eye(5)
    out = model(nodes, adj)
    print("GNN Fraud Detection Output Shape:", out.shape)

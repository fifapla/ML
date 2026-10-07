# Graph Neural Network for Fraud Detection
GNN implementation for detecting suspicious network entities and financial transaction fraud.

**Note:** this uses a hand-rolled, minimal Graph Convolutional layer
(linear transform + adjacency matmul) in plain PyTorch, not the
PyTorch Geometric library - good for showing you understand the core
GCN mechanics, but not a substitute for a production GNN stack (which
would use `torch_geometric` for real sparse graph batching, sampling,
and efficient adjacency handling at scale).

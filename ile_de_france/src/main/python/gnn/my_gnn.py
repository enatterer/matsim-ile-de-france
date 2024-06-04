import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class GNN(nn.Module):
    def __init__(self, policy_input_dim, hidden_dim):
        super(GNN, self).__init__()
        self.conv1 = MyGNNLayer(policy_input_dim, hidden_dim)
        self.conv2 = MyGNNLayer(hidden_dim, hidden_dim)

    def forward(self, edge_index, edge_attr):
        x = self.conv1(edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(edge_index, edge_attr)
        return x    

class MyGNNLayer(MessagePassing):
    def __init__(self, edge_input_dim, hidden_dim):
        super(MyGNNLayer, self).__init__(aggr='mean')  # "mean" aggregation
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, edge_index, edge_attr):
        # Add self-loops to the edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=edge_index.max().item() + 1)
        
        # Initial transformation of edge attributes
        edge_attr = self.edge_mlp(edge_attr)
        
        # Propagate messages
        out = self.propagate(edge_index, edge_attr=edge_attr)
        
        # Final output transformation
        out = self.output_layer(out)
        return out

    def message(self, edge_attr_j):
        # This function constructs messages from edge attributes
        return edge_attr_j

    def update(self, aggr_out):
        # This function updates node embeddings after message passing
        return aggr_out
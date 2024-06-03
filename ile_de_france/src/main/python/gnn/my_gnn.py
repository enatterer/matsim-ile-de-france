import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MyGNN(MessagePassing):
    def __init__(self, policy_input_dim, traffic_input_dim, hidden_dim):
        super(MyGNN, self).__init__(aggr='mean')  # "mean" aggregation.
        self.policy_mlp = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.traffic_mlp = nn.Sequential(
            nn.Linear(traffic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, 1)  # Output a single dimension for traffic flow prediction

    def forward(self, edge_index, policy_edge_attr, traffic_edge_attr):
        # Initial transformation of edge attributes
        policy_hidden = self.policy_mlp(policy_edge_attr)
        traffic_hidden = self.traffic_mlp(traffic_edge_attr)
        
        # Combine the two hidden representations
        combined_hidden = policy_hidden + traffic_hidden
        
        # Message passing
        out = self.propagate(edge_index, combined_hidden=combined_hidden)
        
        # Final output transformation
        output = self.output_layer(out)
        
        return output

    def message(self, combined_hidden_j):
        # This function constructs messages from node j to node i
        return combined_hidden_j

    def update(self, aggr_out):
        # This function updates node embeddings after message passing
        return aggr_out
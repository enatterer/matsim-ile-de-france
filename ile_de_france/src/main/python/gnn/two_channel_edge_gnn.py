import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TwoChannelEdgeGNN(nn.Module):
    def __init__(self, policy_input_dim, traffic_input_dim, hidden_dim):
        super(TwoChannelEdgeGNN, self).__init__()
        self.policy_gnn = nn.Linear(policy_input_dim, hidden_dim)
        self.traffic_gnn = nn.Linear(traffic_input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, traffic_input_dim)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.policy_gnn.weight)
        nn.init.xavier_uniform_(self.traffic_gnn.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.policy_gnn.bias is not None:
            nn.init.zeros_(self.policy_gnn.bias)
        if self.traffic_gnn.bias is not None:
            nn.init.zeros_(self.traffic_gnn.bias)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
            
    def forward(self, policy_features, traffic_features, edge_index):
        traffic_features = traffic_features.view(-1, 1)

        # Apply GNN layers to edge features
        policy_hidden = self.policy_gnn(policy_features)
        traffic_hidden = self.traffic_gnn(traffic_features)
        
        # Combine hidden states
        combined_hidden = policy_hidden + traffic_hidden
        
        # Clip combined hidden values to avoid large values
        combined_hidden = torch.clamp(combined_hidden, min=-1e6, max=1e6)
        
        combined_hidden = torch.spmm(edge_index, combined_hidden) 

        # Check for NaNs after combining
        if torch.isnan(combined_hidden).any():
            print("NaNs found in combined_hidden")
        
        # Pass through output layer
        output = self.output_layer(combined_hidden)
        
        # Check for NaNs in output
        if torch.isnan(output).any():
            print("NaNs found in output")
        
        return output
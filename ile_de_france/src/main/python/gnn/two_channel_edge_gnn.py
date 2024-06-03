import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TwoChannelEdgeGNN(nn.Module):
    def __init__(self, policy_input_dim, traffic_input_dim, hidden_dim):
        super(TwoChannelEdgeGNN, self).__init__()
        self.policy_layer = nn.Linear(policy_input_dim, hidden_dim)
        self.traffic_layer = nn.Linear(traffic_input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, traffic_input_dim) 

    def forward(self, policy_features, traffic_features):
        policy_hidden = self.policy_layer(policy_features)
        traffic_hidden = self.traffic_layer(traffic_features)
        
        # Debug statements to print the shapes
        print("policy_hidden shape:", policy_hidden.shape)
        print("traffic_hidden shape:", traffic_hidden.shape)
        
        combined_hidden = policy_hidden + traffic_hidden
        
        print("combined_hidden shape:", combined_hidden.shape)
        
        # Check for NaNs after combining
        if torch.isnan(combined_hidden).any():
            print("NaNs found in combined_hidden")
        
        output = self.output_layer(combined_hidden)
        
        print("output shape:", output.shape)
        
        # Check for NaNs in output
        if torch.isnan(output).any():
            print("NaNs found in output")
        
        return output
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoChannelGNN(nn.Module):
    def __init__(self, policy_input_dim, traffic_input_dim, hidden_dim):
        super(TwoChannelGNN, self).__init__()
        self.policy_input_dim = policy_input_dim
        self.traffic_input_dim = traffic_input_dim
        self.hidden_dim = hidden_dim
        
        # Define layers for message passing and updates in each channel
        self.policy_message_passing = nn.Linear(policy_input_dim, hidden_dim)
        self.traffic_message_passing = nn.Linear(traffic_input_dim, hidden_dim)
        
        # Define layers for node/edge updates in each channel
        self.policy_update = nn.Linear(hidden_dim, hidden_dim)
        self.traffic_update = nn.Linear(hidden_dim, hidden_dim)
        
        # Define layers for final output
        self.output_layer = nn.Linear(hidden_dim * 2, 1)  # Concatenate representations from both channels
    
    def forward(self, policy_features, traffic_features, adjacency_matrix):
        # Assuming policy_features and traffic_features are of shape (num_edges, input_dim)
        # You may need to reshape or adjust the dimensions of these tensors if necessary
        
        # Message Passing in Policy Channel
        policy_messages = self.policy_message_passing(policy_features)
        policy_aggregated = torch.matmul(adjacency_matrix, policy_messages)  # Aggregating messages from neighbors
        # Ensure that the output dimension of self.policy_message_passing matches the expected dimension for matrix multiplication
        
        # Message Passing in Traffic Flow Channel
        traffic_messages = self.traffic_message_passing(traffic_features) # transforms input traffic features into hidden representation, facilitating the propagation of information through the graph. This then helps the model to learn a more comprehensive and effective representation of the graph data.
        traffic_aggregated = torch.matmul(adjacency_matrix, traffic_messages)  # Aggregating messages from neighbors
        # Ensure that the output dimension of self.traffic_message_passing matches the expected dimension for matrix multiplication
        
        # Node/Edge Updates in Policy Channel
        policy_hidden = F.relu(self.policy_update(policy_aggregated))
        
        # Node/Edge Updates in Traffic Flow Channel
        traffic_hidden = F.relu(self.traffic_update(traffic_aggregated))
        
        # Concatenate representations from both channels
        combined_representation = torch.cat((policy_hidden, traffic_hidden), dim=1)
        
        # Output prediction
        output = self.output_layer(combined_representation)
        return output

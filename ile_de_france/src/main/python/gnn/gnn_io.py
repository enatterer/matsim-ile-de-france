import random

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch_geometric.nn import GCNConv

import torch
import geopandas as gpd
from shapely.geometry import LineString

import pickle


def check_and_replace_inf(policy_data):
    has_inf = False
    inf_sources = {"capacity": 0, "freespeed": 0, "mode": 0}
    
    for i, row in enumerate(policy_data):
        capacity, freespeed, modes = row[0], row[1], row[2]
        
        # Check freespeed for inf values
        if freespeed == float('inf') or freespeed == float('-inf'):
            # print(f"Inf value found in freespeed at row {i}: {freespeed}")
            has_inf = True
            inf_sources["freespeed"] += 1
            row[1] = 1e6 if freespeed == float('inf') else -1e6
    return policy_data, has_inf, inf_sources

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
def get_dataloader(is_train, batch_size, datasets):
    loader = torch.utils.data.DataLoader(dataset=datasets, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


# Define a dictionary to map each mode to an integer
mode_mapping = {
    'bus': 0,
    'car': 1,
    'car_passenger': 2,
    'pt': 3,
    'bus,car,car_passenger': 4,
    'bus,car,car_passenger,pt': 5,
    'car,car_passenger': 6,
    'pt,rail,train': 7,
    'bus,pt': 8,
    'rail': 9,
    'pt,subway': 10,
    'artificial,bus': 11,
    'artificial,rail': 12,
    'artificial,stopFacilityLink,subway': 13,
    'artificial,subway': 14,
    'artificial,stopFacilityLink,tram': 15,
    'artificial,tram': 16,
    'artificial,bus,stopFacilityLink': 17,
    'artificial,funicular,stopFacilityLink': 18,
    'artificial,funicular': 19
}

# Function to encode modes into integer format
def encode_modes(modes):
    return mode_mapping.get(modes, -1)  # Use -1 for any unknown modes

def create_edge_index_and_tensors(gdf):
    # Ensure the GeoDataFrame is projected in a geographic CRS
    gdf = gdf.to_crs(epsg=4326)
    
    # Extract unique nodes (start and end points of LineStrings)
    nodes = {}
    edges = []
    car_volume_counts = []
    capacities = []
    freespeeds = []
    modes_list = []
    node_counter = 0

    for idx, row in gdf.iterrows():
        line = row['geometry']
        car_volume = row['vol_car']  # Extract car volume
        capacity = row['capacity']  # Extract capacity
        freespeed = row['freespeed']  # Extract freeflow speed
        modes = encode_modes(row['modes']) # Extract modes
        if isinstance(line, LineString):
            start_point = line.coords[0]
            end_point = line.coords[-1]
            
            if start_point not in nodes:
                nodes[start_point] = node_counter
                node_counter += 1
            if end_point not in nodes:
                nodes[end_point] = node_counter
                node_counter += 1
            
            start_index = nodes[start_point]
            end_index = nodes[end_point]
            
            edges.append((start_index, end_index))
            car_volume_counts.append(car_volume)  # Store activity count for this edge
            capacities.append(capacity)  # Store capacity for this edge
            freespeeds.append(freespeed)  # Store freespeed speed for this edge
            modes_list.append(modes)  # Store modes for this edge
    
    # Create edge_index tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create car volume tensor
    car_volume_tensor = torch.tensor(car_volume_counts, dtype=torch.float)
    
    # Create policy_tensor
    policy_data = [[cap, fs, mode] for cap, fs, mode in zip(capacities, freespeeds, modes_list)]
    policy_data, has_inf, inf_sources = check_and_replace_inf(policy_data)
    # if has_inf:
    #     print("Inf values were found and replaced in policy_data.")
    #     print("Inf value sources:", inf_sources)
    
    policy_tensor = torch.tensor(policy_data, dtype=torch.float)
    return edge_index, car_volume_tensor, policy_tensor, nodes

# def visualize_data(policy_features, flow_features, title):
#     edges = edge_index.T.tolist()

#     # Create a networkx graph from the edge list
#     G = nx.Graph(edges)

#     # Create a colormap for flow features
#     flow_cmap = plt.cm.Reds     # colormap for flow features

#     # Extract flow features from tensor
#     flow_values = flow_features.tolist()      # flow graph has only one feature atm

#     # Normalize features for colormap mapping
#     flow_min = 0
#     flow_max = 100
#     norm = Normalize(vmin=flow_min, vmax=flow_max)

#     # Draw the graph with separate lines for flow features on each edge and annotations for policy features
#     plt.figure(figsize=(8, 6))
#     pos = nx.spring_layout(G, seed=42)  # Layout for better visualization

#     # Set to store processed edges
#     processed_edges = set()

#     # Draw edges for flow features and annotate with policy features
#     for i, (u, v) in enumerate(edges):
#         # Check if the edge has already been processed
#         if (u, v) not in processed_edges and (v, u) not in processed_edges:
#             flow_color = flow_cmap((flow_values[i] - flow_min) / (flow_max - flow_min))
#             nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=flow_color, width=2, alpha=0.7)

#             # Annotate with policy feature values
#             policy_values_str = ", ".join([f"{int(val)}" for val in policy_features[i]])
#             plt.text((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2, f"({policy_values_str})", fontsize=8, color="black", ha="center", va="center")

#             # Add the edge to the set of processed edges
#             processed_edges.add((u, v))

#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", alpha=0.8)

#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

#     # Add colorbar for flow features
#     flow_sm = plt.cm.ScalarMappable(norm=norm, cmap=flow_cmap)
#     flow_sm.set_array([])
#     plt.colorbar(flow_sm, label="Flow")
#     plt.title(title)
#     # if "Train" in title:
#     #     plt.savefig(f"visualisation/train_data/{title}.png", dpi = 500)
#     # else:
#     #     plt.savefig(f"visualisation/test_data/{title}.png", dpi = 500)
    
def create_edge_adjacency_matrix(edge_index, num_edges):
    indices = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make it undirected
    values = torch.ones(indices.shape[1])
    adj_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([num_edges, num_edges]))
    return adj_matrix
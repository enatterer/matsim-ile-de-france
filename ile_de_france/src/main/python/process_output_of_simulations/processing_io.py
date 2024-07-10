import os
import glob
import gzip
import math
import random
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import shapely.wkt as wkt
from shapely.geometry import Point, LineString, box
from shapely.ops import nearest_points
import lxml.etree as ET
import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import LineGraph

districts = gpd.read_file("../../../../data/visualisation/districts_paris.geojson")

def process_result_dic(result_dic):
    datalist = []
    linegraph_transformation = LineGraph()

    for key, df in result_dic.items():
        if isinstance(df, pd.DataFrame):
            gdf = gpd.GeoDataFrame(df, geometry='geometry')
            gdf.crs = "EPSG:2154"  # Assuming the original CRS is EPSG:2154
            gdf.to_crs("EPSG:4326", inplace=True)
            
            # Create dictionaries for nodes and edges
            nodes = pd.concat([gdf['from_node'], gdf['to_node']]).unique()
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            
            gdf['from_idx'] = gdf['from_node'].map(node_to_idx)
            gdf['to_idx'] = gdf['to_node'].map(node_to_idx)
            
            edges = gdf[['from_idx', 'to_idx']].values
            edge_car_volumes = gdf['vol_car'].values
            capacities = gdf['capacity'].values
            freespeeds = gdf['freespeed'].values  
            lengths = gdf['length'].values  
            modes = gdf['modes'].values
            modes_encoded = np.vectorize(encode_modes)(modes)
            
            edge_positions = np.array([((geom.coords[0][0] + geom.coords[-1][0]) / 2, 
                                        (geom.coords[0][1] + geom.coords[-1][1]) / 2) 
                                       for geom in gdf.geometry])

            # Convert lists to tensors
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_positions_tensor = torch.tensor(edge_positions, dtype=torch.float)
            x = torch.zeros((len(nodes), 1), dtype=torch.float)
            
            # Create Data object
            target_values = torch.tensor(edge_car_volumes, dtype=torch.float).unsqueeze(1)
            data = Data(edge_index=edge_index, x=x, pos=edge_positions_tensor)
            
            # Transform to line graph
            linegraph_data = linegraph_transformation(data)
            
            # Prepare the x for line graph: index and capacity
            linegraph_x = torch.tensor(np.column_stack((capacities, freespeeds, lengths, modes_encoded)), dtype=torch.float)

            linegraph_data.x = linegraph_x
            
            # Target tensor for car volumes
            linegraph_data.y = target_values
            
            if linegraph_data.validate(raise_on_error=True):
                datalist.append(linegraph_data)
            else:
                print("Invalid line graph data")
                
    # Convert dataset to a list of dictionaries
    data_dict_list = [{'x': lg_data.x, 'edge_index': lg_data.edge_index, 'pos': lg_data.pos, 'y': lg_data.y} for lg_data in datalist]
    
    return data_dict_list


# Function to iterate over result_dic and perform the required operations
def analyze_geodataframes(result_dic):
    # Extract the base network data for comparison
    base_gdf = result_dic.get("base_network_no_policies")
    if base_gdf is not None:
        base_vol_car = base_gdf['vol_car'].sum()
        base_gdf_car = base_gdf[base_gdf['modes'].str.contains('car')]
        base_capacity_car = base_gdf_car['capacity'].sum() * 0.05
        # base_freespeed_car = base_gdf_car['freespeed'].sum()

    for policy, gdf in result_dic.items():
        print(f"Policy: {policy}")
        
        # Filter edges where 'modes' contains 'car'
        gdf_car = gdf[gdf['modes'].str.contains('car')]
        total_capacity_car = round(gdf_car['capacity'].sum() * 0.005)
        print(f"Total capacity of edges with 'car' mode: {total_capacity_car}")
        
        # total_freespeed_car = round(gdf_car['freespeed'].sum())
        # print(f"Total freespeed of edges with 'car' mode: {total_freespeed_car}")

        total_vol_car = gdf['vol_car'].sum()
        # print(f"Total 'vol_car': {total_vol_car}")

        if policy != "base_network_no_policies" and base_gdf is not None:
            vol_car_increase = ((total_vol_car - base_vol_car) / base_vol_car) * 100
            capacity_car_increase = ((total_capacity_car - base_capacity_car) / base_capacity_car) * 100
            # freespeed_car_increase = ((total_freespeed_car - base_freespeed_car) / base_freespeed_car) * 100

            print(f"Percentage increase in 'vol_car': {vol_car_increase:.2f}%")
            print(f"Percentage increase in capacity (car edges): {capacity_car_increase:.2f}%")
            # print(f"Percentage increase in freespeed (car edges): {freespeed_car_increase:.2f}%")
        
        
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


# Plotting function
def plot_simulation_output(key, df):
    arrondissement_number = key.replace('policy_', '').replace('_', ' ')
    arrondissement_number_cleaned = arrondissement_number.replace(" ", "_")

    print(f"Policy: {key}")
    
    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:2154")
    gdf = gdf.to_crs(epsg=4326)

    x_min = gdf.total_bounds[0] + 0.05
    y_min = gdf.total_bounds[1] + 0.05
    x_max = gdf.total_bounds[2]
    y_max = gdf.total_bounds[3]
    bbox = box(x_min, y_min, x_max, y_max)
    
    # Filter the network to include only the data within the bounding box
    gdf = gdf[gdf.intersects(bbox)]
    
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    if not gdf[gdf['vol_car'] == 0].empty:
        gdf[gdf['vol_car'] == 0].plot(ax=ax, color='lightgrey', linewidth=0.2, label='Network', zorder=1)
    else:
        print("No geometries with vol_car == 0 to plot.")

    # Plot links with activity_count using the Viridis color scheme and log scale
    # Plot links with vol_car using the coolwarm color scheme and log scale
    gdf.plot(column='vol_car', cmap='coolwarm', linewidth=1.5, ax=ax, legend=True,
             norm=LogNorm(vmin=gdf['vol_car'].min() + 1, vmax=gdf['vol_car'].max()),
             legend_kwds={'label': "Car volume", 'orientation': "vertical", 'shrink': 0.5,
                          'prop': {'family': 'Times New Roman', 'size': 12}})
    
    # districts.plot(ax=ax, facecolor='None', edgecolor='black', linewidth=1, label="Arrondissements", zorder=3)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Customize the plot with Times New Roman font and size 15
    plt.xlabel("Longitude", fontname='Times New Roman', fontsize=15)
    plt.ylabel("Latitude", fontname='Times New Roman', fontsize=15)
    # plt.legend(prop={'family': 'Times New Roman', 'size': 15})
    
    # Customize title and legend labels
    # ax.set_title(arrondissement_number, fontname='Times New Roman', fontsize=15)
    for text in ax.get_legend().get_texts():
        text.set_fontname('Times New Roman')
        text.set_fontsize(15)

    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(10)
        
    plt.savefig("results/" + f"{arrondissement_number_cleaned}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
def create_policy_key_1pct(folder_name):
    # Extract the relevant part of the folder name
    parts = folder_name.split('_')[1:]  # Ignore the first part ('network')
    district_info = '_'.join(parts).replace('d_', '')
    districts = district_info.split('_')
    return f"policy introduced in Arrondissement(s) {', '.join(districts)}"
    
def create_policy_key_1pm(folder_name):
    # Extract the relevant part of the folder name
    base_name = os.path.basename(folder_name)  # Get the base name of the file or folder
    parts = base_name.split('_')[1:]  # Ignore the first part ('network')
    district_info = '_'.join(parts)
    districts = district_info.split('_')
    return f"Policy introduced in Arrondissement(s) {', '.join(districts)}"


def is_single_district(filename):
    return filename.count('_') == 2
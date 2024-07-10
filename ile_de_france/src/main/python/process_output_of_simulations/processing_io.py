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
import re
from matplotlib.colors import TwoSlopeNorm

from shapely.ops import unary_union
from mpl_toolkits.axes_grid1 import make_axes_locatable



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

def plot_simulation_output(df, districts_of_interest: list):
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
    
    # Filter edges based on the "osm:way:highway" column
    highway_types = ["primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link"]
    gdf = gdf[gdf["osm:way:highway"].isin(highway_types)]
    
    # Identify districts 1, 2, 3, 4
    target_districts = districts[districts['c_ar'].isin(districts_of_interest)]
    other_districts = districts[~districts['c_ar'].isin(districts_of_interest)]

    gdf['intersects_target_districts'] = gdf.apply(lambda row: target_districts.intersects(row.geometry).any(), axis=1)

    # Use TwoSlopeNorm for custom normalization
    norm = TwoSlopeNorm(vmin=gdf['vol_car'].min(), vcenter=gdf['vol_car'].median(), vmax=gdf['vol_car'].max())
    

    # Plot the edges that intersect with target districts thicker
    gdf[gdf['intersects_target_districts']].plot(column='vol_car', cmap='coolwarm', linewidth=4, ax=ax, legend=False,
             norm=norm, label = "Higher order roads", zorder=2)
    
    # Plot the other edges
    gdf[~gdf['intersects_target_districts']].plot(column='vol_car', cmap='coolwarm', linewidth=4, ax=ax, legend=False,
             norm=norm, zorder=1)
    
    # Add buffer to target districts to avoid overlapping with edges
    buffered_target_districts = target_districts.copy()
    buffered_target_districts['geometry'] = buffered_target_districts.buffer(0.001)
    # Ensure the buffered_target_districts GeoDataFrame is in the same CRS
    if buffered_target_districts.crs != gdf.crs:
        buffered_target_districts.to_crs(gdf.crs, inplace=True)

    # Create a single outer boundary
    outer_boundary = unary_union(buffered_target_districts.geometry).boundary

    # Plot only the outer boundary
    gpd.GeoSeries(outer_boundary, crs=gdf.crs).plot(ax=ax, edgecolor='black', linewidth=1, label="Arrondissements " + list_to_string(districts_of_interest), zorder=4)

    # ax.set_aspect('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Customize the plot with Times New Roman font and size 15
    plt.xlabel("Longitude", fontname='Times New Roman', fontsize=15)
    plt.ylabel("Latitude", fontname='Times New Roman', fontsize=15)

    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(15)
    ax.legend(prop={'family': 'Times New Roman', 'size': 15})
    
    # Adjust layout to make space for color bar
    # Manually set the position of the main plot axis
    ax.set_position([0.1, 0.1, 0.75, 0.75])

    # Create an axis on the right side for the color bar
    cax = fig.add_axes([0.87, 0.22, 0.03, 0.5])  # Manually position the color bar

    # Create the color bar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, cax=cax)
    
    # Set color bar font properties
    cbar.ax.tick_params(labelsize=15)
    for t in cbar.ax.get_yticklabels():
        t.set_fontname('Times New Roman')
    cbar.ax.yaxis.label.set_fontname('Times New Roman')
    cbar.ax.yaxis.label.set_size(15)
    cbar.set_label('Car volume: Difference to base case', fontname='Times New Roman', fontsize=15)
    plt.savefig("results/difference_to_policies_in_zones_" + list_to_string(districts_of_interest, "_"), bbox_inches='tight')
    plt.show()
    
def list_to_string(integers, delimiter=', '):
    """
    Converts a list of integers into a string, with each integer separated by the specified delimiter.

    Parameters:
    integers (list of int): The list of integers to convert.
    delimiter (str): The delimiter to use between integers in the string.

    Returns:
    str: A string representation of the list of integers.
    """
    return delimiter.join(map(str, integers))

def get_subdirs(directory: str):
    full_path = '../../../../data/' + directory + "/"
    subdirs_pattern = os.path.join(full_path, 'output_seed_*')
    subdirs_list = list(set(glob.glob(subdirs_pattern)))
    subdirs_list.sort()
    return subdirs_list

# Function to read and convert CSV.GZ to GeoDataFrame
def read_network_data(file_path):
    if os.path.exists(file_path):
        # Read the CSV file with the correct delimiter
        df = pd.read_csv(file_path, delimiter=';')
        # Convert the 'geometry' column to actual geometrical data
        df['geometry'] = df['geometry'].apply(wkt.loads)
        
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        return gdf
    else:
        return None
    

def extract_numbers(path):
    name = path.split('/')[-1]
    # Use regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', name)
    # Convert the list of numbers to a set of integers
    return set(map(int, numbers))

def create_dic(subdir: str):
    result_dic = {}
    for s in subdir:
        # print(f'Accessing folder: {s}')
        random_seed = extract_numbers(s)
        output_links = s + "/output_links.csv.gz"
        gdf = read_network_data(output_links)
        if gdf is not None:
            result_dic[str(random_seed)] = gdf
    return result_dic

def compute_average_or_median_geodataframe(geodataframes, column_name, is_mean: bool = True):
    """
    Compute the average GeoDataFrame from a list of GeoDataFrames for a specified column.
    
    Parameters:
    geodataframes (list of GeoDataFrames): List containing GeoDataFrames
    column_name (str): The column name for which to compute the average
    
    Returns:
    GeoDataFrame: A new GeoDataFrame with the average values for the specified column
    """
    # Create a copy of the first GeoDataFrame to use as the base
    average_gdf = geodataframes[0].copy()
    
    # Extract the specified column values from all GeoDataFrames
    column_values = np.array([gdf[column_name].values for gdf in geodataframes])
    
    if (is_mean):
    # Calculate the average values for the specified column
        column_average = np.mean(column_values, axis=0)
    else:
        column_average = np.median(column_values, axis=0)

    # Assign the average values to the new GeoDataFrame
    average_gdf[column_name] = column_average
    
    return average_gdf


def compute_difference_geodataframe(gdf_to_substract_from, gdf_to_substract, column_name):
    """
    Compute the difference of a specified column between two GeoDataFrames.
    
    Parameters:
    gdf1 (GeoDataFrame): The first GeoDataFrame
    gdf2 (GeoDataFrame): The second GeoDataFrame
    column_name (str): The column name for which to compute the difference
    
    Returns:
    GeoDataFrame: A new GeoDataFrame with the differences for the specified column
    """
    # Ensure the two GeoDataFrames have the same shape
    if gdf_to_substract_from.shape != gdf_to_substract.shape:
        raise ValueError("GeoDataFrames must have the same shape")

    # Ensure the two GeoDataFrames have the same indices
    if not gdf_to_substract_from.index.equals(gdf_to_substract.index):
        raise ValueError("GeoDataFrames must have the same indices")
    
    # Ensure the two GeoDataFrames have the same geometries
    if not gdf_to_substract_from.geometry.equals(gdf_to_substract.geometry):
        raise ValueError("GeoDataFrames must have the same geometries")
    
    # Create a copy of the first GeoDataFrame to use as the base for the difference GeoDataFrame
    difference_gdf = gdf_to_substract_from.copy()

    # Compute the difference for the specified column
    difference_gdf[column_name] = gdf_to_substract_from[column_name] - gdf_to_substract[column_name]

    return difference_gdf

def remove_columns(gdf_with_correct_columns, gdf_to_be_adapted):
    """
    Remove columns from gdf1 that are not present in gdf2.
    
    Parameters:
    gdf1 (GeoDataFrame): The GeoDataFrame from which columns will be removed
    gdf2 (GeoDataFrame): The GeoDataFrame that provides the column template
    
    Returns:
    GeoDataFrame: A new GeoDataFrame with only the columns present in gdf2
    """
    columns_to_keep = gdf_with_correct_columns.columns
    gdf1_filtered = gdf_to_be_adapted[columns_to_keep]
    return gdf1_filtered

def extend_geodataframe(gdf_base, gdf_to_extend, column_name):
    """
    Extend a GeoDataFrame by adding a column from another GeoDataFrame.
    
    Parameters:
    gdf_base (GeoDataFrame): The GeoDataFrame containing the column to add
    gdf_to_extend (GeoDataFrame): The GeoDataFrame to be extended
    column_name (str): The column name to add to gdf_to_extend
    
    Returns:
    GeoDataFrame: A new GeoDataFrame with the column added
    """
    # Ensure the column exists in the base GeoDataFrame
    if column_name not in gdf_base.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the base GeoDataFrame")
    
    # Create a copy of the GeoDataFrame to be extended
    extended_gdf = gdf_to_extend.copy()
    
    # Add the column from the base GeoDataFrame
    extended_gdf[column_name] = gdf_base[column_name]
    
    return extended_gdf
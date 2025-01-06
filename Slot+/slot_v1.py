#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 02:59:58 2024

@author: johnnynienstedt
"""

#
# Early Attempts to Quantify Arm Slot Effects
# Johnny Nienstedt 12/29/24
#

#
# This is the third leg of 4S Pitching. The goal of this analysis is to
# quantify the value that pitchers gain/lose purely from their arm slot.
# Currently this entails movement deviation, but in the future I may include
# arm slot uniqueness, arm slot variance, or hidden ball percentage. 
#

import math
import hdbscan
import pybaseball
import numpy as np
import pandas as pd
import datashader as ds
import scipy.stats as stats
import matplotlib.pyplot as plt
import datashader.transfer_functions as tf
from tqdm import tqdm
from scipy.stats import iqr
from bokeh.palettes import Category20, Spectral
from sklearn.preprocessing import StandardScaler



'''
###############################################################################
############################# Import & Clean Data #############################
###############################################################################
'''

# import classified pitch data
classified_pitch_data = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Data/classified_pitch_data.csv')

# add xRV
strike_types = ['swinging_strike_blocked', 'called_strike', 'swinging_strike', 'foul_tip', 'missed_bunt', 'bunt_foul_tip']
ball_types = ['ball', 'blocked_ball', 'pitchout']
foul_types = ['foul', 'foul_bunt', 'foul_pitchout']

classified_pitch_data['xRV'] = np.where(classified_pitch_data['description'] == 'hit_into_play', 0.192 - 0.6708*classified_pitch_data['estimated_woba_using_speedangle'], np.nan)
classified_pitch_data['xRV'] = np.where(classified_pitch_data['description'].isin(strike_types), 0.08575, classified_pitch_data['xRV'])
classified_pitch_data['xRV'] = np.where(classified_pitch_data['description'].isin(ball_types), -0.05697, classified_pitch_data['xRV'])
classified_pitch_data['xRV'] = np.where(classified_pitch_data['description'].isin(foul_types), 0.03537, classified_pitch_data['xRV'])
classified_pitch_data['xRV'] = np.where(classified_pitch_data['description'] == 'hit_by_pitch', -0.38164, classified_pitch_data['xRV'])

# select columns of interest
shape_data = classified_pitch_data[['pitcher', 'player_name', 'game_year', 'true_pitch_type', 'release_pos_x', 'release_pos_y', 'release_pos_z', 'release_speed', 'pfx_x', 'pfx_z', 'VAA', 'HAA', 'xRV']]

arm_angles_21 = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Data/pitcher_arm_angles_2021.csv')
arm_angles_22 = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Data/pitcher_arm_angles_2022.csv')
arm_angles_23 = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Data/pitcher_arm_angles_2023.csv')
arm_angles_24 = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Data/pitcher_arm_angles_2024.csv')

# add year column
arm_angles_21['game_year'] = 2021
arm_angles_22['game_year'] = 2022
arm_angles_23['game_year'] = 2023
arm_angles_24['game_year'] = 2024

arm_angles = pd.concat([arm_angles_21, arm_angles_22, arm_angles_23, arm_angles_24], axis=0)
arm_angles = arm_angles[['pitcher', 'game_year', 'n_pitches', 'ball_angle']]

# merge
slot_data = pd.merge(shape_data, arm_angles, on=['pitcher', 'game_year'], how='outer')
slot_data.dropna(subset='ball_angle', inplace=True)
slot_data['n_pitches'] = slot_data['n_pitches'].astype(int)

# calculate effective velo
slot_data['eff_velo'] = round(slot_data['release_speed'] * slot_data['release_pos_y'].mean() / slot_data['release_pos_y'], 1)
slot_data[['VAA', 'HAA']] = round(slot_data[['VAA', 'HAA']], 1)
slot_data = slot_data[['pitcher', 'player_name', 'game_year', 'n_pitches', 'true_pitch_type', 'release_pos_x', 'release_pos_z', 'ball_angle', 'eff_velo', 'pfx_x', 'pfx_z', 'VAA', 'HAA', 'xRV']]



'''
###############################################################################
############################## Make Slot Clusters #############################
###############################################################################
'''

def create_arm_slot_clusters(data, sample_size=100000, min_cluster_size=100, min_samples=10):
    """
    Create clusters of similar arm slots using HDBSCAN, optimized for large datasets.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the arm slot variables
    sample_size (int): Number of samples to use for initial clustering
    min_cluster_size (int): Minimum size for a cluster
    min_samples (int): Number of samples in a neighborhood for a point to be core
    
    Returns:
    tuple: (clustered_data, clusterer, scaler)
    """
    print("Starting clustering process...")
    
    # Extract arm slot features
    slot_features = ['release_pos_x', 'release_pos_z', 'ball_angle']
    
    # Take a random sample for initial model fitting
    np.random.seed(42)
    sample_indices = np.random.choice(data.index, size=sample_size, replace=False)
    sample_data = data.loc[sample_indices, slot_features].copy()
    
    print(f"Took sample of {sample_size} points for initial clustering")
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    scaler.fit(data[slot_features])  # Fit on all data to get true distribution
    sample_scaled = scaler.transform(sample_data)
    
    # Create and fit the clusterer
    print("Fitting HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.1,
        metric='euclidean',
        algorithm='best',
        core_dist_n_jobs=-1  # Use all available CPU cores
    )
    
    # Fit the model on the sample
    clusterer.fit(sample_scaled)
    
    print("Predicting clusters for full dataset...")
    
    # Process the full dataset in chunks
    chunk_size = 100000
    all_labels = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        chunk_scaled = scaler.transform(chunk[slot_features])
        # Simply use the predict method
        labels = clusterer.fit_predict(chunk_scaled)
        all_labels.extend(labels)
        
        if i % 500000 == 0:
            print(f"Processed {i} rows...")
    
    # Add cluster labels to the original data
    result_data = data.copy()
    result_data['slot_cluster'] = all_labels
    
    # Print cluster statistics
    n_clusters = len(set(all_labels)) - (1 if -1 in all_labels else 0)
    print(f"\nNumber of clusters: {n_clusters}")
    print("\nCluster sizes:")
    for cluster in sorted(set(all_labels)):
        size = sum(np.array(all_labels) == cluster)
        if cluster == -1:
            print(f"Noise points: {size} ({size/len(all_labels):.1%})")
        else:
            print(f"Cluster {cluster}: {size} ({size/len(all_labels):.1%})")
    
    return result_data, clusterer, scaler

def plot_clusters_datashader(data, width=800, height=600):
    """
    Create a visualization of the arm slot clusters using datashader.
    Handles large datasets efficiently.
    """
    # Convert cluster labels to category
    plot_data = data.copy()
    plot_data['slot_cluster'] = plot_data['slot_cluster'].astype('category')
    
    # Number of unique clusters
    n_clusters = plot_data['slot_cluster'].nunique()
    
    # Generate a categorical colormap with enough colors
    # Use Category20, or combine palettes for more colors
    if n_clusters <= 20:
        palette = Category20[n_clusters]
    elif n_clusters <= 256:
        palette = Category20[20] + Spectral[min(256, n_clusters - 20)]
    else:
        raise ValueError(f"Too many clusters ({n_clusters}) for available palettes.")
    
    # Create canvas
    cvs = ds.Canvas(plot_width=width, plot_height=height)
    
    # Aggregate points
    agg = cvs.points(plot_data, 'release_pos_x', 'release_pos_z', ds.count_cat('slot_cluster'))
    
    # Create image
    img = tf.shade(agg, color_key=dict(zip(plot_data['slot_cluster'].cat.categories, palette)))
    
    return img

# make clusters (~2 min)
clustered_data, clusterer, scaler = create_arm_slot_clusters(slot_data)

# visualization
img = plot_clusters_datashader(clustered_data)
plt.imshow(img.to_pil(), aspect='auto')
plt.axis('off')  # Optional: Hide axes for better visualization
plt.show()



'''
###############################################################################
############################ Make Effectiveness Map ###########################
###############################################################################
'''



# find zeros for each cluster
x_means = dict(round(clustered_data.groupby(['slot_cluster', 'true_pitch_type']).mean(numeric_only=True)['pfx_x'], 1))
z_means = dict(round(clustered_data.groupby(['slot_cluster', 'true_pitch_type']).mean(numeric_only=True)['pfx_z'], 1))

def set_x_means(cluster, true_pitch_type):
    if pd.isna(true_pitch_type):
        return np.nan
    return x_means[cluster, true_pitch_type]
def set_z_means(cluster, true_pitch_type):
    if pd.isna(true_pitch_type):
        return np.nan
    return z_means[cluster, true_pitch_type]

clustered_data['x_zero'] = clustered_data.apply(lambda row: set_x_means(row['slot_cluster'], row['true_pitch_type']), axis=1)
clustered_data['z_zero'] = clustered_data.apply(lambda row: set_z_means(row['slot_cluster'], row['true_pitch_type']), axis=1)

clustered_data['x_rel'] = clustered_data['pfx_x'] - clustered_data['x_zero']
clustered_data['z_rel'] = clustered_data['pfx_z'] - clustered_data['z_zero']


def create_shape_effectiveness_map(data, pitch_type, effectiveness_metric='xRV'):
    """
    Creates a shape-space effectiveness map for a given pitch type.
    
    Parameters:
    data (pd.DataFrame): Pitch data including movement, approach angles, and effectiveness
    pitch_type (str): The type of pitch to analyze
    effectiveness_metric (str): Column name for the effectiveness measure
    
    Returns:
    tuple: (normalized_data, effectiveness_map)
    """
    # Filter for the specific pitch type
    filtered_data = data[data['true_pitch_type'] == pitch_type].copy()
    
    # normalize effectiveness
    filtered_data[effectiveness_metric] = (filtered_data[effectiveness_metric] - filtered_data[effectiveness_metric].mean()).copy()
    
    shape_params = ['x', 'z']
    # Create effectiveness map
    # We'll bin the normalized data and calculate mean effectiveness in each bin
    bins = 5  # number of bins for each dimension
    
    def get_bin_edges(series, n_bins):
        """
        Create bin edges concentrated around the center of the distribution.
        Uses linear spacing between p20 and p80, with catch-all bins for extremes.
        """
        p20 = np.percentile(series, 20)
        p80 = np.percentile(series, 80)
        
        # Create linearly spaced bins between p20 and p80
        center_edges = np.linspace(p20, p80, n_bins - 1)  # -1 to account for the extremes
        
        # Add the extreme edges
        edges = np.concatenate([
            [series.min()],  # Lowest bin catches all low outliers
            center_edges,
            [series.max()]   # Highest bin catches all high outliers
        ])
        
        return np.unique(edges)  # Ensure monotonicity by removing any duplicates
    
    # Create bin edges for each parameter
    bin_edges = {
        param: get_bin_edges(filtered_data[f'{param}_rel'], bins)
        for param in shape_params
    }
    
    # Assign bin numbers to each pitch
    for param in shape_params:
        filtered_data[f'{param}_bin'] = np.digitize(
            filtered_data[f'{param}_rel'],
            bin_edges[param]
        )  - 1# subtract 1 to make 0-based
    
    # Calculate effectiveness for each bin
    effectiveness_map = filtered_data.groupby(
        [f'{param}_bin' for param in shape_params]
    )[effectiveness_metric].agg(['mean', 'count']).reset_index()
    
    # remove outliers
    effectiveness_map = effectiveness_map.query('x_bin < @bins and z_bin < @bins').copy()
    
    # assign bin edges
    for xbin in range(bins):
        for zbin in range(bins):
            effectiveness_map.loc[effectiveness_map['x_bin'] == xbin, 'pfx_x_min'] = bin_edges['x'][xbin]
            effectiveness_map.loc[effectiveness_map['x_bin'] == xbin, 'pfx_x_max'] = bin_edges['x'][xbin + 1]
            effectiveness_map.loc[effectiveness_map['z_bin'] == zbin, 'pfx_z_min'] = bin_edges['z'][zbin]
            effectiveness_map.loc[effectiveness_map['z_bin'] == zbin, 'pfx_z_max'] = bin_edges['z'][zbin + 1]

    return effectiveness_map



def plot_movement_effectiveness_combined(effectiveness_map, pitch_type):
    """
    Create a smoothed heatmap of pitch effectiveness based on actual movement values.
    Uses sample-size weighted smoothing and ensures the weighted average is zero.
    
    Parameters:
    -----------
    effectiveness_map : DataFrame
        DataFrame containing the bin information and effectiveness values.
        Must include a 'count' column with the number of pitches in each bin.
    pitch_type : str
        Name of the pitch type for the title
    min_samples : int
        Minimum number of samples required in each bin
    smoothing_sigma : float
        Standard deviation for Gaussian smoothing. Higher values = more smoothing
    """
    from scipy.ndimage import gaussian_filter
    import numpy as np
    
    df = effectiveness_map.copy()
    df['mean'] = 1000 * df['mean']
    
    n_samples = df['count'].sum()
    sample_size = round(math.log10(n_samples))
    smoothing_sigma = 7.5/sample_size
    
    # First, create a regular grid
    x_edges = np.unique(np.concatenate([df['pfx_x_min'], df['pfx_x_max']]))
    z_edges = np.unique(np.concatenate([df['pfx_z_min'], df['pfx_z_max']]))
    
    # Get the bin centers for plotting
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    
    # Create empty grids for values, counts, and masks
    grid_values = np.full((len(z_centers), len(x_centers)), np.nan)
    grid_counts = np.zeros_like(grid_values)
    mask = np.full_like(grid_values, False, dtype=bool)
    
    # Fill the grids with values and counts
    for _, row in df.iterrows():
        x_idx = np.where(np.isclose(x_centers, (row['pfx_x_min'] + row['pfx_x_max']) / 2))[0][0]
        z_idx = np.where(np.isclose(z_centers, (row['pfx_z_min'] + row['pfx_z_max']) / 2))[0][0]
        grid_values[z_idx, x_idx] = row['mean']
        grid_counts[z_idx, x_idx] = row['count']
        mask[z_idx, x_idx] = True
    
    # Create weighted data for smoothing
    weighted_data = np.copy(grid_values)
    weighted_data[~mask] = 0
    weighted_data = weighted_data * grid_counts
    
    # Smooth both the weighted data and the counts
    smooth_weighted_data = gaussian_filter(weighted_data, sigma=smoothing_sigma)
    smooth_counts = gaussian_filter(grid_counts, sigma=smoothing_sigma)
    
    # Calculate the smoothed values, normalized by counts
    with np.errstate(divide='ignore', invalid='ignore'):
        grid_smooth = np.where(smooth_counts > 5,  # Threshold to avoid division by small numbers
                             smooth_weighted_data / smooth_counts,
                             np.nan)
    
    # Center the values to ensure weighted average is zero
    valid_mask = ~np.isnan(grid_smooth)
    if np.any(valid_mask):
        # Calculate current weighted average
        current_weighted_avg = np.average(
            grid_smooth[valid_mask],
            weights=smooth_counts[valid_mask]
        )
        # Subtract the weighted average to center at zero
        grid_smooth[valid_mask] -= current_weighted_avg
        
        # Verify the new weighted average is approximately zero
        new_weighted_avg = np.average(
            grid_smooth[valid_mask],
            weights=smooth_counts[valid_mask]
        )
        assert abs(new_weighted_avg) < 1e-5, "Weighted average is not zero after centering"
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set fixed colormap limits
    vmin, vmax = -20, 20  # Fixed run value limits
    norm = plt.Normalize(vmin, vmax)
    
    # Create the heatmap using pcolormesh
    mesh = ax.pcolormesh(
        x_edges, 
        z_edges, 
        grid_smooth,
        cmap='bwr',
        norm=norm,
        shading='flat'
    )
    
    # Add labels and title
    min_samples = df['count'].min()
    value = round(np.sum(np.abs(grid_smooth))/2)

    ax.set_title(f'{pitch_type} xRV by Arm Slot Deviation (Min. {min_samples} samples per bin)\n\u00B1{value} runs per 1000 pitches available due to slot deviation.')
    ax.set_xlabel('Unexpected Horizontal Movement (inches from slot-derived median, pitcher POV)')
    ax.set_ylabel('Unexpected Vertical Movement (inches from median)')
    
    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add colorbar with fixed limits
    fig.colorbar(mesh, ax=ax, label='xRV per 1000 pitches', shrink= 0.7)
    
    plt.tight_layout()
    plt.show()
    
    return grid_smooth    

effectiveness_map = create_shape_effectiveness_map(clustered_data, 'Riding Fastball')
plot_movement_effectiveness_combined(effectiveness_map, 'Riding Fastball')

eff_dict = {}
for pitch_type in tqdm(classified_pitch_data['true_pitch_type'].unique()):

    if pd.isna(pitch_type):
        continue
    
    eff_dict[pitch_type] = create_shape_effectiveness_map(clustered_data, 
                                                          pitch_type=pitch_type, 
                                                          effectiveness_metric='xRV')

'''
###############################################################################
########################## Predicted Shape by Cluster #########################
###############################################################################
'''






'''
###############################################################################
################################ Grade Pitchers ###############################
###############################################################################
'''

def calculate_slot_rv(clustered_data, eff_dict, classified_pitch_data):
    """
    Vectorized calculation of slot_rv for pitch data.
    
    Parameters:
    -----------
    clustered_data : pd.DataFrame
        DataFrame containing pitch data with columns: slot_cluster, true_pitch_type, x_rel, z_rel
    eff_dict : dict
        Dictionary containing efficiency DataFrames for each pitch type
    classified_pitch_data : pd.DataFrame
        DataFrame containing classified pitch data
    
    Returns:
    --------
    pd.DataFrame
        Updated clustered_data with slot_rv column
    """
    # Create a copy to avoid modifying the original
    result_data = clustered_data.copy()
    result_data['slot_rv'] = 0.0
    
    # Process each pitch type in parallel
    for pitch_type in tqdm(classified_pitch_data['true_pitch_type'].unique()):
        if pd.isna(pitch_type):
            continue
            
        # Get relevant data for this pitch type
        eff_df = eff_dict[pitch_type]
        mask = (result_data['true_pitch_type'] == pitch_type) & (result_data['slot_cluster'] != -1)
        pitch_data = result_data[mask]
        
        if len(pitch_data) == 0:
            continue
        
        # Create bins once for this pitch type
        xbins = np.append(eff_df['pfx_x_min'].unique(), eff_df['pfx_x_max'].max())
        xbins[0] = -1000
        xbins[-1] = 1000
        zbins = np.append(eff_df['pfx_z_min'].unique(), eff_df['pfx_z_max'].max())
        zbins[0] = -1000
        zbins[-1] = 1000
        
        # Vectorized bin calculation
        xbin = np.digitize(pitch_data['x_rel'].values, xbins) - 1
        zbin = np.digitize(pitch_data['z_rel'].values, zbins) - 1
        
        # Create a lookup table from eff_df
        lookup_df = eff_df.set_index(['x_bin', 'z_bin'])['mean']
        
        # Vectorized lookup using index
        try:
            rv_values = lookup_df.loc[list(zip(xbin, zbin))].values
            result_data.loc[mask, 'slot_rv'] = rv_values
        except KeyError:
            # Handle any missing bin combinations
            for x, z, idx in zip(xbin, zbin, pitch_data.index):
                try:
                    result_data.at[idx, 'slot_rv'] = lookup_df.loc[(x, z)]
                except KeyError:
                    continue
    
    return result_data

grade_data = calculate_slot_rv(clustered_data, eff_dict, classified_pitch_data)

grade_data['slot_rv'] = grade_data['slot_rv']/grade_data['n_pitches']

slot_grades = grade_data[grade_data['game_year'] == 2024].groupby(['pitcher', 'player_name']).sum(numeric_only=True)['slot_rv'].reset_index()
sim_mean = slot_grades['slot_rv'].mean()
sim_std = slot_grades['slot_rv'].std()

slot_grades['Slot+'] = ((slot_grades['slot_rv'] - sim_mean)/sim_std + 10)*10
slot_grades.to_csv('slot_grades.csv')



















pitcher_results = pybaseball.pitching_stats(2021, 2024, qual = 50, ind = 1)

def plot_yearly_variance(pitcher_results, predictor, result):
    
    # Shift the dataframe to create year pairs
    df1 = pitcher_results[['Name', 'Season', predictor, result]].copy()
    df2 = pitcher_results[['Name', 'Season', predictor, result]].copy()
    
    df1['next_year'] = df1['Season'] + 1
    merged = pd.merge(df1, df2, left_on=['Name', 'next_year'], right_on=['Name', 'Season'],
                      suffixes=('_year1', '_year2'))

    # Create scatter plots for each year pair
    plt.figure(figsize=(10, 6))
    
    x = merged[f'{predictor}_year1']
    y = merged[f'{result}_year2']
    
    plt.scatter(x, y, s=10, alpha=0.7)
    
    # Linear regression for trendline
    m, b, r, p, std_err = stats.linregress(x, y)
    plt.plot(x, m*x+b, '--k')
    
    # Set plot details
    plt.title(f'{predictor} vs Next Year {result}')
    plt.xlabel(f'Year 1 {predictor}')
    plt.ylabel(f'Year 2 {result}')
    
    left = x.min()
    top = y.max() * 0.9
    
    plt.text(left, top, f'$R^2$ = {round(r**2, 2)}')
    plt.show()

# Example usage
plot_yearly_variance(pitcher_results, 'botCmd', 'Location+')

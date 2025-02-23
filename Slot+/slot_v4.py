#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 02:59:58 2024

@author: johnnynienstedt
"""

#
# Slot+
# Johnny Nienstedt 12/29/24
#

# 
# The third leg of my 4S Pitching model. 
#
# The purpose of this script is to quantify the value that pitchers gain/lose 
# purely from their arm slot. The main driver of this is movement deviation
# from expectation, but I have included a slot rarity factor as well to reward
# pitchers with unique arm slots.
#

# changes from V3:
    # smooth cluster means

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import get_ipython
from matplotlib.ticker import MaxNLocator



'''
###############################################################################
################################# Import Data #################################
###############################################################################
'''

# import data
classified_pitch_data = pd.read_csv('classified_pitch_data.csv')



'''
###############################################################################
############################### Slot Deviation ################################
###############################################################################
'''

# make clusters
clustered_data = classified_pitch_data.dropna(subset='arm_angle').copy()
clustered_data['slot_cluster'] = 5 * round(clustered_data['arm_angle']/5)

#
# Make Effectiveness Map
#

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

# smooth these expectations
def smooth_zeros(data):
    
    zeroed_data = data.copy()
    
    for pitch_type in tqdm(clustered_data['true_pitch_type'].unique()):
        
        if pd.isna(pitch_type): 
            continue
    
        else:
            # Get the data for the specific pitch type
            df = zeroed_data.query('true_pitch_type == @pitch_type')[['slot_cluster', 'x_zero', 'z_zero']]
            
            # Calculate weighted means and counts for each slot
            grouped = df.groupby('slot_cluster').agg({
                'x_zero': 'mean',
                'z_zero': 'mean',
                'slot_cluster': 'count'  # This gives us the count of pitches per slot
            }).rename(columns={'slot_cluster': 'count'})
            
            slot = grouped.index.values
            x = grouped['x_zero'].values
            z = grouped['z_zero'].values
            weights = grouped['count'].values
            
            # for x, use quadratic fit
            best_fit = np.polynomial.polynomial.polyfit(slot, x, 2, w=weights)
            predicted_zeros = np.polynomial.polynomial.polyval(df['slot_cluster'], best_fit)
            zeroed_data.loc[zeroed_data['true_pitch_type'] == pitch_type, 'x_zero'] = round(predicted_zeros, 1)
            
            # for z, use linear fit
            best_fit = np.polynomial.polynomial.polyfit(slot, z, 1, w=weights)
            predicted_zeros = np.polynomial.polynomial.polyval(df['slot_cluster'], best_fit)
            zeroed_data.loc[zeroed_data['true_pitch_type'] == pitch_type, 'z_zero'] = round(predicted_zeros, 1)
    
    return zeroed_data

rel_data = smooth_zeros(clustered_data)
rel_data['x_rel'] = rel_data['pfx_x'] - rel_data['x_zero']
rel_data['z_rel'] = rel_data['pfx_z'] - rel_data['z_zero']

def create_shape_effectiveness_map(data, pitch_type, effectiveness_metric='woba_value'):
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
    filtered_data[effectiveness_metric] = (filtered_data[effectiveness_metric] - filtered_data[effectiveness_metric].mean())
    
    shape_params = ['x', 'z']
    # Create effectiveness map
    # We'll bin the normalized data and calculate mean effectiveness in each bin
    bins = 7  # number of bins for each dimension
    
    def get_bin_edges(series, n_bins):
        """
        Create bin edges concentrated around the center of the distribution.
        Uses linear spacing between p20 and p80, with catch-all bins for extremes.
        """
        plo = np.percentile(series, 10)
        phi = np.percentile(series, 90)
        
        # Create linearly spaced bins between p20 and p80
        center_edges = np.linspace(plo, phi, n_bins - 1)  # -1 to account for the extremes
        
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
    
    if effectiveness_metric == 'woba_value':
        effectiveness_map['mean'] = -0.0378*effectiveness_map['mean']
    
    # assign bin edges
    for xbin in range(bins):
        for zbin in range(bins):
            effectiveness_map.loc[effectiveness_map['x_bin'] == xbin, 'pfx_x_min'] = bin_edges['x'][xbin]
            effectiveness_map.loc[effectiveness_map['x_bin'] == xbin, 'pfx_x_max'] = bin_edges['x'][xbin + 1]
            effectiveness_map.loc[effectiveness_map['z_bin'] == zbin, 'pfx_z_min'] = bin_edges['z'][zbin]
            effectiveness_map.loc[effectiveness_map['z_bin'] == zbin, 'pfx_z_max'] = bin_edges['z'][zbin + 1]

    return effectiveness_map

def plot_movement_effectiveness_combined(effectiveness_map, pitch_type, pitcher=None, year=None):
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
    df['mean'] = 3000 * df['mean']
    
    n_samples = df['count'].sum()
    sample_size = round(math.log10(n_samples))
    smoothing_sigma = 7.5/sample_size
    
    # First, create a regular grid
    x_edges = np.unique(np.concatenate([df['pfx_x_min'], df['pfx_x_max']]))
    z_edges = np.unique(np.concatenate([df['pfx_z_min'], df['pfx_z_max']]))
    
    shortest_edge = np.min(np.abs([df['pfx_x_min'], df['pfx_x_max'], df['pfx_z_min'], df['pfx_z_max']]))*12
    
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
    fig, ax = plt.subplots(figsize=(7, 5.33))
    
    # Set fixed colormap limits
    vmin, vmax = -10, 10  # Fixed run value limits
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
    value = round((np.max(grid_smooth) - np.min(grid_smooth))/2)

    if pitcher:        
        hand = classified_pitch_data[classified_pitch_data['player_name'] == pitcher].p_throws.values[0]
        pfirst = pitcher.split(', ')[1]
        plast = pitcher.split(', ')[0]
        ax.set_title('Expected Run Value by Arm Slot Deviation for\n' + year + ' ' +  pfirst + ' ' + plast + ' (' + hand + f'HP), {pitch_type}', fontsize = 16)
        ax.set_xlabel('Unexpected Horizontal Break (Pitcher POV)')
        ax.set_ylabel('Unexpected Vertical Break (in.)')
    else:
        ax.set_title(f'{pitch_type} xRV by Arm Slot Deviation (Min. {min_samples} samples per bin)\n\u00B1{value} runs per 3000 pitches available due to slot deviation.')
        ax.set_xlabel('Unexpected Horizontal Break (inches from slot-derived median, pitcher POV)')
        ax.set_ylabel('Unexpected Vertical Break (inches from median)')
    
    ax.set_xlim((-shortest_edge, shortest_edge))
    ax.set_ylim((-shortest_edge, shortest_edge))
    
    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add colorbar with fixed limits
    if pitcher:
        pass
    else:
        fig.colorbar(mesh, ax=ax, label='xRV per 3000 pitches', shrink= 0.7, ticks = [-5, 0, 5])
    
    ax.set_aspect(1)
    
    if pitcher:
        return fig, ax 
    else:
        plt.tight_layout()
        plt.show()

eff_dict = {}
for pitch_type in tqdm(classified_pitch_data['true_pitch_type'].unique()):

    if pd.isna(pitch_type):
        continue
    
    eff_dict[pitch_type] = create_shape_effectiveness_map(rel_data, 
                                                          pitch_type=pitch_type, 
                                                          effectiveness_metric='woba_value')



'''
###############################################################################
################################# Slot Rarity #################################
###############################################################################
'''

# what percentage of pitches are thrown within 5º of your arm angle?
slot_frequencies = rel_data['slot_cluster'].value_counts(normalize=True)
rel_data['slot_rarity'] = rel_data['slot_cluster'].map(slot_frequencies)



'''
###############################################################################
################################ Grade Pitchers ###############################
###############################################################################
'''

def calculate_diff_rv(rel_data, classified_pitch_data):
    """
    Vectorized calculation of diff_rv for pitch data.
    
    Parameters:
    -----------
    rel_data : pd.DataFrame
        DataFrame containing pitch data with columns: slot_cluster, true_pitch_type, x_rel, z_rel
    eff_dict : dict
        Dictionary containing efficiency DataFrames for each pitch type
    classified_pitch_data : pd.DataFrame
        DataFrame containing classified pitch data
    
    Returns:
    --------
    pd.DataFrame
        Updated rel_data with diff_rv column
    """

    result_data = rel_data.copy()
    result_data['diff_rv'] = 0.0
    
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
        rv_values = lookup_df.loc[list(zip(xbin, zbin))].values
        result_data.loc[mask, 'diff_rv'] = rv_values

    
    return result_data

grade_data = calculate_diff_rv(rel_data, classified_pitch_data)

def normalize(data, desired_mean, desired_std=None, dtype=float):
    
    m = data.mean()
    s = data.std()
    
    if desired_std:
        normalized_data = (((data - m) / s) * desired_std + desired_mean).astype(dtype)
    else:
        normalized_data = ((data - m) + desired_mean).astype(dtype)
    
    return normalized_data

slot_grades = grade_data[grade_data['game_year'] > 2022].groupby(['game_year', 'pitcher', 'player_name']).mean(numeric_only=True)[['diff_rv', 'slot_rarity']].dropna().reset_index()
# slot_grades.to_csv('slot_grades.csv')



'''
###############################################################################
############################### Visualize Grades ##############################
###############################################################################
'''

def visualize_grade(pitcher, pitch_type, year):
    
    ipython = get_ipython()
    ipython.run_line_magic('matplotlib', 'qt')
    
    hand = classified_pitch_data[classified_pitch_data['player_name'] == pitcher].p_throws.values[0]
    
    df = rel_data.query('player_name == @pitcher and true_pitch_type == @pitch_type and game_year == @year')
    
    if len(df) == 0:
        print('\nNo data for this combination of pitcher, pitch type, and year.')
        return
    
    n_pitches = 1000

    xRV = round(grade_data.query('player_name == @pitcher and true_pitch_type == @pitch_type and game_year == @year')['diff_rv'].mean()*n_pitches, 1)
    
    if xRV > 0:
        xRV = '+' + str(round(xRV, 1))
    else:
        xRV = str(round(xRV, 1))
    
    x0 = df['x_zero'].mean()
    z0 = df['z_zero'].mean()

    def x_abs(x_rel):
        return x_rel + x0
    def x_rel(x_abs):
        return x_abs + x0
    def z_abs(z_rel):
        return z_rel + z0
    def z_rel(z_abs):
        return z_abs + z0

    fig, ax = plot_movement_effectiveness_combined(eff_dict[pitch_type], pitch_type, pitcher, '2024')
    
    xax = ax.secondary_xaxis('top', functions=(x_abs, x_rel))
    xax.set_xlabel('Actual Horizontal Break (in.)')
    zax = ax.secondary_yaxis('right', functions=(z_abs, z_rel))
    zax.set_ylabel('Actual Inducued Vertical Break (in.)')
    
    x = df['x_rel']
    z = df['z_rel']
    x_mean = x.median()
    x_std = x.std()
    z_mean = z.median()
    z_std = z.std()
    
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = x_mean + x_std * np.cos(theta)
    ellipse_z = z_mean + z_std * np.sin(theta)
    
    # Plot the ellipse
    ax.fill(ellipse_x, ellipse_z, alpha=0.5, fc='k')
    ax.scatter(x_mean, z_mean, s = 50, color = 'k', label = xRV + ' xRV\nper 1000 pitches')
    ax.legend(fontsize=14)
        
    # ensure plot is large enough
    xlims = ax.get_xlim()
    zlims = ax.get_ylim()
    
    if x_mean < xlims[0]:
        expansion_factor = (xlims[1] - x_mean) / (xlims[1] - xlims[0])
        ax.set_xlim((x_mean, xlims[1]))
        ax.set_ylim((zlims[0] * expansion_factor, zlims[1] * expansion_factor))
    if x_mean > xlims[1]:
        expansion_factor = (x_mean - xlims[0]) / (xlims[1] - xlims[0])
        ax.set_xlim((xlims[0], x_mean))
        ax.set_ylim((zlims[0] * expansion_factor, zlims[1] * expansion_factor/2))
    if z_mean < zlims[0]:
        expansion_factor = (zlims[1] - z_mean) / (zlims[1] - zlims[0])
        ax.set_ylim((z_mean, zlims[1]))
        ax.set_xlim((xlims[0] * expansion_factor, xlims[1] * expansion_factor))
    if z_mean > zlims[1]:
        expansion_factor = (z_mean - zlims[0]) / (zlims[1] - zlims[0])
        ax.set_ylim((zlims[0], z_mean))
        ax.set_xlim((xlims[0] * expansion_factor, xlims[1] * expansion_factor))

    if hand == 'L':
        plt.gca().invert_xaxis()
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    xax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    zax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
    
    ax.set_aspect(1)
    plt.tight_layout()
    plt.show()
    
# effectiveness_map = create_shape_effectiveness_map(rel_data, 'Riding Fastball', 'woba_value')
# plot_movement_effectiveness_combined(effectiveness_map, 'Riding Fastball')

# visualize_grade('Richards, Trevor', 'Riding Fastball', 2024)
# visualize_grade('Richards, Trevor', 'Velo-Based Changeup', 2024)

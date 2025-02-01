#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:51:35 2024

@author: johnnynienstedt
"""

#
# Spot+
# Johnny Nienstedt 12/27/24
#

# 
# The second leg of my 4S Pitching model. 
#
# The purpose of this script is to grade pitchers' ability to throw their 
# pitches in the correct locations. Instead of using estimated target
# location to evaluate command, this model simply evaluates each pitch's 
# location for expected run value (based on count, handedness, and pitch type).
#

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from numba import jit
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter
from scipy.spatial.distance import pdist



'''
###############################################################################
################################# Import Data #################################
###############################################################################
'''

# heatmaps from my SEAGER project
league_heatmaps = np.load('../Data/league_heatmaps.npy')
classified_pitch_data = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Data/classified_pitch_data.csv')



'''
###############################################################################
############################### Location Value ################################
###############################################################################
'''

# reformat pitch types to match my old syntax
pitch_names = np.array([
    'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider',
    'Two-Plane Slider', 'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 
    'Movement-Based Changeup', 'Velo-Based Changeup', 'Knuckleball'])

pitch_classes = {'fastball': ['Riding Fastball', 'Fastball'],
                 'sinker': ['Sinker'],
                 'cutter': ['Cutter'],
                 'slider': ['Gyro Slider', 'Two-Plane Slider', 'Sweeper'],
                 'curveball': ['Slurve', 'Curveball', 'Slow Curve'],
                 'changeup': ['Movement-Based Changeup', 'Velo-Based Changeup', 'Knuckleball']
                 }

classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['fastball']), 0, np.nan)
classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['sinker']), 1, classified_pitch_data['pitch_id'])
classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['cutter']), 2, classified_pitch_data['pitch_id'])
classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['slider']), 3, classified_pitch_data['pitch_id'])
classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['curveball']), 4, classified_pitch_data['pitch_id'])
classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['changeup']), 5, classified_pitch_data['pitch_id'])
classified_pitch_data = classified_pitch_data.dropna(subset = 'pitch_id')

# pixel resolution of heatmaps
XLIST = np.linspace(-15/13.5, 15/13.5, 30) + 1/27
ZLIST = np.linspace(14/12, (14+32)/12, 35) + 1/27

@jit(nopython=True)
def find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or 
        abs(value - array[idx-1]) < abs(value - array[idx])):
        return idx-1
    return idx

def loc_rv(df, league_heatmaps):

    # Convert inputs to arrays for vectorized operations
    balls = pd.to_numeric(df['balls'], errors='coerce').astype(np.int8)
    strikes = pd.to_numeric(df['strikes'], errors='coerce').astype(np.int8)
    platoon = pd.to_numeric(df['platoon'], errors='coerce').astype(np.int8)
    t = pd.to_numeric(df['pitch_id'], errors='coerce').astype(np.int8)
    px = pd.to_numeric(df['plate_x'], errors='coerce')
    pz = pd.to_numeric(df['plate_z'], errors='coerce')
    
    # Create mask for valid inputs
    valid_mask = ((balls >= 0) & (balls <= 3) & 
                 (strikes >= 0) & (strikes <= 2) & 
                 (platoon >= 0) & (platoon <= 1) &
                 px.notna() & pz.notna())
    
    # Initialize results array
    result = np.full(len(df), np.nan)
    
    # Process only valid rows
    if not valid_mask.any():
        return result
        
    # Find nearest grid points using vectorized operations
    x_idx = np.array([find_nearest_idx(XLIST, x) for x in px[valid_mask]])
    z_idx = np.array([find_nearest_idx(ZLIST, z) for z in pz[valid_mask]])
    
    # Get run values for valid entries
    result[valid_mask] = league_heatmaps[0, 
                                         balls[valid_mask],
                                         strikes[valid_mask], 
                                         t[valid_mask],
                                         platoon[valid_mask],
                                         5,
                                         x_idx,
                                         z_idx]
    
    return result

# Usage
classified_pitch_data['loc_rv'] = loc_rv(classified_pitch_data, league_heatmaps)



'''
###############################################################################
############################## Release Variance ###############################
###############################################################################
'''

#
# On the repertoire level this is actually a sequence effect, but it's easier
# to just calculate them all at once.
#

def calculate_all_variances(classified_pitch_data, min_pitches=100, min_pitch_ratio=0.03, years=[2023, 2024]):
    
    # Filter data once for the specified year
    year_df = classified_pitch_data[classified_pitch_data['game_year'].isin(years)]
    
    # Initialize DataFrame for results
    pitcher_years = year_df[['game_year', 'pitcher']].drop_duplicates()
    slot_variance_df = pd.DataFrame(
        index=pd.MultiIndex.from_frame(pitcher_years[['game_year', 'pitcher']]),
        columns=['Name', 'release_variance', 'repertoire_variance', 'pbp_variance']
    )
    
    # Group by pitcher and year
    pitcher_groups = year_df.groupby(['game_year', 'pitcher'])
    
    for _, row in pitcher_years.iterrows():
        
        try:
            df = pitcher_groups.get_group((row['game_year'], row['pitcher']))
        except KeyError:
            continue

        if len(df) < min_pitches:
            continue
        
        # Set name
        last, first = df['player_name'].iloc[0].split(', ')
        slot_variance_df.loc[(row['game_year'], row['pitcher']), 'Name'] = f"{first} {last}"
        
        # Calculate overall release variance using combined std of horizontal and vertical angles
        release_points = df[['release_angle_h', 'release_angle_v']].to_numpy()
        total_std = np.sum(np.std(release_points, axis=0))
        slot_variance_df.loc[(row['game_year'], row['pitcher']), 'release_variance'] = total_std
        
        # Group by pitch type
        pitch_groups = df.groupby('true_pitch_type')
        pitch_type_stats = []
        
        # Calculate per-pitch type statistics
        for pitch_type, p_df in pitch_groups:
            if len(p_df)/len(df) < min_pitch_ratio:
                continue
            
            # Store mean release point and std for each pitch type
            mean_release = p_df[['release_angle_h', 'release_angle_v']].mean().values
            std_release = np.sum(p_df[['release_angle_h', 'release_angle_v']].std().values)
            pitch_type_stats.append({
                'pitch_type': pitch_type,
                'mean_release': mean_release,
                'std': std_release,
                'count': len(p_df)
            })
        
        if pitch_type_stats:
            # Calculate repertoire variance (between pitch types)
            # Use average pairwise distance between mean release points
            mean_points = np.array([stat['mean_release'] for stat in pitch_type_stats])
            if len(mean_points) > 1:
                pairwise_distances = pdist(mean_points)
                repertoire_variance = np.mean(pairwise_distances)
            else:
                repertoire_variance = 0
            slot_variance_df.loc[(row['game_year'], row['pitcher']), 'repertoire_variance'] = repertoire_variance
            
            # Calculate pbp variance as weighted average of per-pitch type stds
            total_pitches = sum(stat['count'] for stat in pitch_type_stats)
            weighted_std = sum(stat['std'] * stat['count'] for stat in pitch_type_stats) / total_pitches
            slot_variance_df.loc[(row['game_year'], row['pitcher']), 'pbp_variance'] = weighted_std
    
    return slot_variance_df

slot_variance_df = calculate_all_variances(classified_pitch_data)

slot_variance_df[['release_variance', 'repertoire_variance', 'pbp_variance']] = slot_variance_df[['release_variance', 'repertoire_variance', 'pbp_variance']].astype(float)
slot_variance_df = slot_variance_df.reset_index()

slot_data = pd.merge(classified_pitch_data, slot_variance_df, on=['pitcher', 'game_year'], how='left')
slot_data = slot_data.dropna(subset = 'release_variance')



'''
###############################################################################
############################### Grade Pitchers ################################
###############################################################################
'''

# group by pitcher and year
loc_grades = slot_data.groupby(['pitcher', 'player_name', 'game_year'])[['loc_rv', 'release_variance', 'repertoire_variance', 'pbp_variance']].mean(numeric_only=True).reset_index()
counts = slot_data.groupby(['pitcher', 'player_name', 'game_year']).size().reset_index()

loc_grades['count'] = counts[0]
loc_grades['mean_loc'] = round(-(loc_grades['loc_rv'] - loc_grades['loc_rv'].mean()), 6)
loc_grades['loc_rv'] = loc_grades['mean_loc']*loc_grades['count']

# bayesian regression with strongly informative normal prior
mu = 0
var = 1
n = loc_grades['count']
x = loc_grades['mean_loc']
s = 63**2
loc_grades['proj_loc'] = round((mu/var + n*x/s)/(1/var + n/s), 6)
loc_grades['proj_var'] = round(1/(1/var + n/s), 6)

# final grades
spot_grades = loc_grades[['game_year', 'pitcher', 'player_name', 'count', 'loc_rv', 'proj_loc', 'release_variance', 'repertoire_variance', 'pbp_variance']]
# spot_grades.to_csv('Spot+/spot_grades.csv')



'''
###############################################################################
################################## Visualize ##################################
###############################################################################
'''

def yoy(df, colname, q, verbose=False):
    
    filtered_results = df[df['count'] > q].reset_index().sort_values(by='pitcher')
    filtered_results.dropna(subset = [colname], inplace=True)
    
    rows = []
    for i in range(len(filtered_results) - 1):
        if filtered_results['pitcher'][i + 1] == filtered_results['pitcher'][i]:
            c1 = filtered_results[colname][i]
            c2 = filtered_results[colname][i + 1]
            c3 = filtered_results['count'][i]
            rows.append({'Year1': c1,
                          'Year2': c2,
                          'N_P': c3})
            
            
    pairs = pd.DataFrame(rows)
    
    
    # make plot
    x = pairs.Year1
    y = pairs.Year2
    
    m, b, r, p, std_err = stats.linregress(x, y)
    
    if verbose:
        fig, ax = plt.subplots()
        plt.scatter(x, y, s=3)
        x1 = np.linspace(x.min(), x.max(), len(x))
        plt.plot(x1, m*x1+b, '--k')
        ax.set_title("YOY Command (Pitcher Level, min. " + str(q) + " Pitches Per Season)")
        ax.set_xlabel('Year 1 Location RV (per 2000 pitches)')
        ax.set_ylabel('Year 2 Location RV (per 2000 pitches)')
            
        # text
        y_limits = ax.get_ylim()
        bot = y_limits[0] + 0.1*(y_limits[1] - y_limits[0])
        top = y_limits[1] - 0.1*(y_limits[1] - y_limits[0])
        x_limits = ax.get_xlim()
        left = x_limits[0] + 0.1*(x_limits[1] - x_limits[0])
        right = x_limits[1] - 0.1*(x_limits[1] - x_limits[0])
        
        if r > 0:
            ax.text(left, top, '$R^2$ = ' + str(round(r**2, 2)), ha='left', va='center', fontsize=10)
        else:
            ax.text(right, top, '$R^2$ = ' + str(round(r**2, 2)), ha='right', va='center', fontsize=10)
        
        ax.text((left + right)/2, bot, 'Data via Baseball Savant 2021-2024', ha='center', va='top', fontsize=8)
        
        # percent if necessary
        if '%' in colname:
            if (y.max() - y.min()) > 0.1:
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100*y:.0f}%'))
                plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{100*x:.0f}%'))
            else:
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100*y:.1f}%'))
                plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{100*x:.1f}%'))
        
        plt.show()

    return r**2
    

# r = []
# for i in np.arange(0, 3000, 100):
#     r.append(yoy(loc_grades, 'spot_rv', i))
    
# plt.figure()
# plt.plot(np.arange(0, 3000, 100), r)
# plt.ylim(bottom=0)
# plt.title('Spot RV Year to Year Correlation')
# plt.xlabel('Minimum Number of Pitches Thrown in Each Year')
# plt.ylabel('$R^2$ Correlation between Year 1 and Year 2 Spot RV')
# plt.show()


# plot prior against actual distribution
x = np.linspace(-20, 20, 1000)
prior = norm.pdf(x, mu, var)

# plot posteriors
def plot_command(df, pitchers, years, include_prior = True):
    
    plt.figure()
    
    if include_prior:
        plt.plot(x, prior, label = 'Prior')
    
    if type(pitchers) == list:
        
        for i, pitcher in enumerate(pitchers):
            
            name = pitcher.split(', ')[1] + ' ' +  pitcher.split(', ')[0]
            
            try:
                year = years[i]
                index = df.query('player_name == @pitcher and game_year == @year').index[0]
                
            except IndexError:
                print('\n', pitcher, 'not found.')
                
            p = norm.pdf(x, df.loc[index, 'proj_loc']*3000, df.loc[index, 'proj_var'])
            
            plt.plot(x, p, label = str(year) + ' ' + name)
            
    elif type(pitchers) == str:
        
        name = pitchers.split(', ')[1] + ' ' + pitchers.split(', ')[0]
        
        try:
            index = df.query('player_name == @pitchers and game_year == @years').index[0]
            
        except IndexError:
            print('\n', pitchers, 'not found.')
            
        p = norm.pdf(x, df.loc[index, 'proj_loc']*3000, df.loc[index, 'proj_var'])
        
        plt.plot(x, p, label = str(years) + ' ' + name)

    plt.xlim(-10, 10)
    plt.ylim(bottom=0)
    plt.title('Posterior Spot+ Distributions')
    plt.xlabel('Projected Spot Runs (Location RV per 3000 pitches)')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper left')
    plt.show()

# plot_command(loc_grades, ['Kirby, George', 'McKenzie, Triston'], [2024, 2024], include_prior = True)

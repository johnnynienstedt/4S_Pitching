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

# Changes from V1:
    # retrieved swing data for each count separately

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
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
classified_pitch_data = pd.read_csv('classified_pitch_data.csv')



'''
###############################################################################
################################ Make Heatmaps ################################
###############################################################################
'''

# get league data for all years
def get_league_data(pitch_data):
    
    ###########################################################################
    ############################# Get League Data #############################
    ###########################################################################
    
    print()
    print('Gathering League Data')
    print()
    
    # RE24 values
    global ball_rv, strike_rv
    ball_rv = np.zeros([4, 3])
    strike_rv = np.zeros([4, 3])
    
    ball_rv[0,0] = 0.032
    ball_rv[1,0] = 0.088
    ball_rv[2,0] = 0.143
    ball_rv[3,0] = 0.051
    ball_rv[0,1] = 0.024
    ball_rv[1,1] = 0.048
    ball_rv[2,1] = 0.064
    ball_rv[3,1] = 0.168
    ball_rv[0,2] = 0.021
    ball_rv[1,2] = 0.038
    ball_rv[2,2] = 0.085
    ball_rv[3,2] = 0.234
    strike_rv[0,0] = -0.037
    strike_rv[1,0] = -0.035
    strike_rv[2,0] = -0.062
    strike_rv[3,0] = -0.117
    strike_rv[0,1] = -0.051
    strike_rv[1,1] = -0.054
    strike_rv[2,1] = -0.069
    strike_rv[3,1] = -0.066
    strike_rv[0,2] = -0.150
    strike_rv[1,2] = -0.171
    strike_rv[2,2] = -0.209
    strike_rv[3,2] = -0.294
    
    
    global swing_types, take_types, bunt_types, pitch_types
    
    swing_types = ['hit_into_play', 'foul', 'swinging_strike', 'foul_tip', 
                    'swinging_strike_blocked', 'swinging_pitchout',
                    'foul_pitchout']
    take_types = ['ball', 'called_strike', 'blocked_ball', 'hit_by_pitch', 
                  'pitchout']
    bunt_types = ['missed_bunt', 'foul_bunt', 'foul_tip_bunt']
    
    pitch_types = [['FF'], ['SI'], ['FC'], ['SL', 'ST'], ['CU', 'KC', 'CS', 'SV'], ['CH', 'FO', 'FS', 'SC'], pitch_data['pitch_type'].unique()]
    
    z_flip = [None, 3, 2, 1, 6, 5, 4, 9, 8, 7, None, 12, 11, 14, 13]


    # flip x-axis for left-handed hitters
    pitch_data.loc[pitch_data['stand'] == 'L', 'plate_x'] = -pitch_data.loc[pitch_data['stand'] == 'L', 'plate_x']
    pitch_data.loc[pitch_data['stand'] == 'L', 'zone'] = pitch_data.loc[pitch_data['stand'] == 'L', 'zone'].apply(lambda z: z_flip[int(z)])


    # initialize data frame and array
    league_rv = np.empty([2,4,3,7,2,13])
    # indices are:
        # data type (swing_rate = 0, swing_rv = 1)
        # balls
        # strikes
        # pitch type (ff = 0, si = 1, ct = 2, sl = 3, cu = 4, ch = 5, all = 6)
        # handedness (same = 0, different = 1)
        # MLBAM zone (normalized to RHB)

    #
    # evaluate pitches in range of MLBAM zones (2.2 ft wide x 3 ft tall)
    #            
    
    # get swing RV based on contact%, whiff%, and xWOBACON       
    for j in tqdm(range(13)):
        
        if j < 9: zone = j + 1
        else: zone = j + 2
        
        # pitches in this zone
        z_pitches = pitch_data[pitch_data.zone == zone]
        
        for t in range(7):
            
            # pitches of this type
            pitch_group = pitch_types[t]
            t_pitches = z_pitches[z_pitches.pitch_type.isin(pitch_group)]
                
            for h in range(2):
                
                # pitches in this platoon matchup
                if h == 0:
                    h_pitches = t_pitches[t_pitches.stand == t_pitches.p_throws]
                        
                if h == 1:
                    h_pitches = t_pitches[t_pitches.stand != t_pitches.p_throws]
            
                # calculated run value for swings, based on RE24
                for s in range(3):
                    
                    s_pitches = h_pitches[h_pitches['strikes'] == s]
                    
                    for b in range(4):
                
                        pitches = s_pitches[s_pitches['balls'] == b]
                        
                        n = len(pitches)
                        
                        # pitches swung at
                        swings = pitches[pitches.description.isin(swing_types)]
                        
                        # number of swings
                        n_swings = len(swings)
                        
                        if n != 0:
                            swing_rate = n_swings/n
                        else:
                            swing_rate = 0
                        
                        league_rv[0,b,s,t,h,j] = swing_rate
                        
                        # contact & foul ball percentage
                        if n_swings != 0:
                            contact = sum(swings.description == 'hit_into_play')/n_swings
                            foul = sum(swings.description == 'foul')/n_swings
                            whiff = 1 - contact - foul
                        else:
                            contact, foul, whiff = 0, 0, 0
                                    
                        # observed run value on balls in play
                        if contact != 0:
                            xwobacon = swings.loc[swings.description == 'hit_into_play','estimated_woba_using_speedangle'].fillna(0).mean()
                            bip_rv = 0.6679*xwobacon - 0.192
                        else:
                            bip_rv  = 0
                        
                        if s == 2:
                            swing_rv = (contact*bip_rv + whiff*strike_rv[b,s])
                        else:
                            swing_rv = (contact*bip_rv + (whiff + foul)*strike_rv[b,s])
                        
                        league_rv[1,b,s,t,h,j] = swing_rv
    
    ###########################################################################
    ############################## Make Heatmaps ##############################
    ###########################################################################
    
    print()
    print()
    print('Making Heatmaps')
    print()
    
    # strike zone dimensions
    zone_height = 35
    zone_width = 30
    
    # number of iterations for numerical solution
    n_iter = 5
    
    # create zones
    x0, y0 = 0, 0
    x1, y1 = 5, 6
    x15, y15 = 9, 10
    x2, y2 = 12, 14
    x25, y25 = 15, 18
    x3, y3 = 18, 22
    x35, y35 = 21, 26
    x4, y4 = 25, 30
    x5, y5 = 30, 35
    
    # Initialize arrays
    league_heatmaps = np.empty([n_iter + 1, 5, 4, 3, 7, 2, zone_width, zone_height])
    # 1st dimension is for intermediate heatmaps
    # 2nd dimension is now:
        # data type (xrv = 0, swing_rate = 1, swing_rv = 2, trv = 3, cs = 4)
    # will be reshaped to (5, 4, 3, 7, 2, n, zw, zh)
         
    #
    # merge zone data to make swing heatmaps
    #   
 
    # initial conditions
    z1 = league_rv[:,:,:,:,:,0]
    z2 = league_rv[:,:,:,:,:,1]
    z3 = league_rv[:,:,:,:,:,2]
    z4 = league_rv[:,:,:,:,:,3]
    z5 = league_rv[:,:,:,:,:,4]
    z6 = league_rv[:,:,:,:,:,5]
    z7 = league_rv[:,:,:,:,:,6]
    z8 = league_rv[:,:,:,:,:,7]
    z9 = league_rv[:,:,:,:,:,8]
    z11 = league_rv[:,:,:,:,:,9]
    z12 = league_rv[:,:,:,:,:,10]
    z13 = league_rv[:,:,:,:,:,11]
    z14 = league_rv[:,:,:,:,:,12]
    
    league_map = np.empty([zone_width, zone_height, 2, 4, 3, 7, 2])
    
    # initial condition set/reset function
    def set_conds(rv_map, reset = False):
        
        if not reset:
            # Set the initial conditions by zone
            rv_map[x1:x2, y3:y4] = z1
            rv_map[x2:x3, y3:y4] = z2
            rv_map[x3:x4, y3:y4] = z3
            rv_map[x1:x2, y2:y3] = z4
            rv_map[x2:x3, y2:y3] = z5
            rv_map[x3:x4, y2:y3] = z6
            rv_map[x1:x2, y1:y2] = z7
            rv_map[x2:x3, y1:y2] = z8
            rv_map[x3:x4, y1:y2] = z9
            rv_map[x0:x1, y25:y5] = z11
            rv_map[x1:x25, y4:y5] = z11
            rv_map[x25:x5, y4:y5] = z12
            rv_map[x4:x5, y25:y5] = z12
            rv_map[x0:x1, y0:y25] = z13
            rv_map[x1:x25, y0:y1] = z13
            rv_map[x25:x5, y0:y1] = z14
            rv_map[x4:x5, y0:y25] = z14
        
        # reset boundary conditions and maintain mean
        if reset: 
            
            # boundary conditions
            rv_map[x0:x0,y25:y5] = z11
            rv_map[x0:x25,y5:y5] = z11
            
            rv_map[x25:x5,y5:y5] = z12
            rv_map[x5:x5,y25:y5] = z12
            
            rv_map[x0:x25,y0:y0] = z13
            rv_map[x0:x0,y0:y25] = z13
            
            rv_map[x5:x5,y0:y25] = z14
            rv_map[x25:x5,y0:y0] = z14
            
            # maintain middle of each zone
            rv_map[x15:x15,y35:y35] = z1
            rv_map[x25:x25,y35:y35] = z2
            rv_map[x35:x35,y35:y35] = z3
            rv_map[x15:x15,y25:y25] = z4
            rv_map[x25:x25,y25:y25] = z5
            rv_map[x35:x35,y25:y25] = z6
            rv_map[x15:x15,y15:y15] = z7
            rv_map[x25:x25,y15:y15] = z8
            rv_map[x35:x35,y15:y15] = z9
        
        return rv_map
    
    
    # set initial conditions
    league_map = np.empty([zone_width, zone_height, 2, 4, 3, 7, 2])
    league_map = set_conds(league_map)
    league_heatmaps[0,1:3] = np.transpose(league_map, (2,3,4,5,6,0,1))
    
    # make heatmap using np.roll method
    for n in range(n_iter):
        league_map = 0.25*(np.roll(league_map, zone_height - 1, axis = 1) + np.roll(league_map, 1 - zone_height, axis = 1) + np.roll(league_map, zone_width - 1, axis = 0) + np.roll(league_map, 1 - zone_width, axis = 0))
        league_map = set_conds(league_map, reset = True)       
        league_heatmaps[n+1, 1:3] = np.transpose(league_map, (2,3,4,5,6,0,1))

    league_heatmaps = np.transpose(league_heatmaps, (1,2,3,4,5,0,6,7))


    # next get called strike % (more granular than swing rv)
    for X in tqdm(range(30)):
        x = round((X - 15)/13.5, 2)
        xx = round((X - 14)/13.5, 2)
        for Z in range(35):
            z = round((Z*32/35 + 14)/12, 2)
            zz = round(((Z + 1)*32/35 + 14)/12, 2)
            
            pitches = pitch_data[(pitch_data.plate_x >= x) &
                                  (pitch_data.plate_x < xx) &
                                  (pitch_data.plate_z >= z) &
                                  (pitch_data.plate_z < zz)]
            
            # pitches taken
            takes = pitches[pitches.description.isin(take_types)]
            
            # percentage of taken pitches called stikes
            n_takes = len(takes)
            cs = sum(takes.description == 'called_strike')/n_takes
            
            league_heatmaps[4,:,:,:,:,n_iter,X,Z] = cs

            # different values of TRV for each count
            for s in range(3):
                for b in range(4):
                    
                    # and calculate TRV using RE24
                    take_rv = cs*strike_rv[b,s] + (1 - cs)*ball_rv[b,s]
                    
                    # append to array
                    league_heatmaps[3,b,s,:,:,n_iter,X,Z] = take_rv
                    
                        
    # now calculate expected RV
    swing_rate = league_heatmaps[1]
    take_rate = 1 - swing_rate
    swing_rv = league_heatmaps[2]
    take_rv = league_heatmaps[3]
    
    xrv = swing_rate*swing_rv + take_rate*take_rv
    league_heatmaps[0] = xrv
                  
    return league_heatmaps

league_heatmaps = get_league_data(classified_pitch_data)



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
#     r.append(yoy(loc_grades, 'loc_rv', i))
    
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

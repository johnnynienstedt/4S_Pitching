#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:40:42 2025

@author: johnnynienstedt
"""

#
# Data Prep for 4S+
#
# Johnny Nienstedt 1/27/25
#

import statcast_pitches
import numpy as np
import polars as pl
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm



'''
###############################################################################
############################# Import & Clean Data #############################
###############################################################################
'''

# load all pitches from 2015-present
pitches_lf = statcast_pitches.load()

# filter to get 2021-2024 data
polars_data = (pitches_lf
                    .filter(pl.col("game_date").dt.year() >= 2021)
                    .collect())

all_pitch_data = polars_data.to_pandas()

# remove deprecated/unnecessary columns (reduce size for computational speed)
good_cols = ['game_date', 'game_year', 'game_pk', 'home_team', 'inning',
             'at_bat_number','pitch_number', 'balls', 'strikes', 'player_name', 
             'pitcher', 'batter', 'stand', 'p_throws', 'pitch_type', 
             'description','events', 'release_speed', 'release_pos_x', 
             'release_pos_y', 'release_pos_z', 'release_extension', 
             'release_spin_rate', 'spin_axis', 'zone', 'plate_x', 'plate_z', 
             'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 
             'sz_bot', 'launch_speed', 'launch_angle', 'launch_speed_angle', 
             'estimated_woba_using_speedangle', 'woba_value', 'delta_run_exp',
             'arm_angle']

nececssary_cols = ['game_date', 'game_year', 'game_pk', 'home_team', 'inning',
                   'at_bat_number','pitch_number', 'balls', 'strikes', 
                   'player_name', 'pitcher', 'batter', 'stand', 'p_throws', 
                   'description', 'release_speed', 'release_pos_x', 
                   'release_pos_y', 'release_pos_z', 'release_extension',
                   'zone', 'plate_x', 'plate_z', 'pfx_x', 'pfx_z', 'vx0', 
                   'vy0', 'vz0', 'ax', 'ay', 'az', 'delta_run_exp']

clean_data = all_pitch_data[good_cols].dropna(subset = nececssary_cols).reset_index()

# remove pitchouts ans position players pitching
clean_data = clean_data[~clean_data['pitch_type'].isin(['PO', 'FA'])]

# select pitcher years with at least 100 pitches thrown
pitch_data = clean_data.groupby(['game_year', 'pitcher']).filter(lambda x: len(x) >= 100)

# flip axis for RHP so that +HB = arm side, -HB = glove side
mirror_cols = ['release_pos_x', 'plate_x', 'pfx_x', 'vx0', 'ax']
pitch_data.loc[pitch_data['p_throws'] == 'R', mirror_cols] = -pitch_data.loc[pitch_data['p_throws'] == 'R', mirror_cols]
pitch_data['spin_axis'] = np.where(pitch_data['p_throws'] == 'R', 360 - pitch_data['spin_axis'], pitch_data['spin_axis'])

# scale HB and iVB
pitch_data[['pfx_x', 'pfx_z']] = np.round(pitch_data[['pfx_x', 'pfx_z']]*12)

# get altitude data
altitude_df = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Data/park_altitude.csv')



'''
###############################################################################
########################## Calculate More Parameters ##########################
###############################################################################
'''

# platoon rate
pitch_data['platoon'] = np.where(pitch_data['p_throws'] == pitch_data['stand'], 0, 1)

# arm angle_adj. spin axis
pitch_data['adj_spin_axis'] = pitch_data['spin_axis'] - pitch_data['arm_angle'] - 90
pitch_data['adj_spin_axis'] = np.where(pitch_data['adj_spin_axis'] < 0, 360 + pitch_data['adj_spin_axis'], pitch_data['adj_spin_axis'])

# vertical and horizontal release angle
vx0 = pitch_data['vx0']
vy0 = pitch_data['vy0']
vz0 = pitch_data['vz0']
ax = pitch_data['ax']
ay = pitch_data['ay']
az = pitch_data['az']
rx = pitch_data['release_pos_x']
ry = pitch_data['release_pos_y']
rz = pitch_data['release_pos_z']
velo = pitch_data['release_speed']
y0 = 50
yf = 17/12

theta_z0 = -np.arctan(vz0/vy0)*180/np.pi
theta_x0 = -np.arctan(vx0/vy0)*180/np.pi
pitch_data['release_angle_v'] = round(theta_z0, 2)
pitch_data['release_angle_h'] = round(theta_x0, 2)

# vertical and horizontal approach angle
vyf = -np.sqrt(vy0**2- (2 * ay * (y0 - yf)))
t = (vyf - vy0)/ay
vzf = vz0 + (az*t)
vxf = vx0 + (ax*t)

theta_zf = -np.arctan(vzf/vyf)*180/np.pi
theta_xf = -np.arctan(vxf/vyf)*180/np.pi
pitch_data['VAA'] = round(theta_zf, 2)
pitch_data['HAA'] = round(theta_xf, 2)

# break angle
delta_y = ry - yf
delta_theta_z = theta_z0 - theta_zf
delta_theta_x = theta_x0 - theta_xf
pitch_data['break_angle_v'] = round(delta_theta_z, 2)
pitch_data['break_angle_h'] = round(delta_theta_x, 2)

# adjust VAA/HAA for location
x = pitch_data['plate_z']
y = pitch_data['VAA']
m, b, r, p, std_err = stats.linregress(x, y)
pitch_data['loc_adj_VAA'] = pitch_data['VAA'] - (m * pitch_data['VAA'] + b)

x = pitch_data['plate_x']
y = pitch_data['HAA']
m, b, r, p, std_err = stats.linregress(x, y)
pitch_data['loc_adj_HAA'] = pitch_data['HAA'] - (m * pitch_data['HAA'] + b)



'''
###############################################################################
############################# Assign Pitch Result #############################
###############################################################################
'''

# 0 for same handedness, 1 for different
pitch_data['platoon'] = np.where(pitch_data['p_throws'] == pitch_data['stand'], 0, 1) 

# add non-bip swing outcomes
swstr_types = ['swinging_strike_blocked', 'swinging_strike', 'foul_tip']
pitch_data['launch_speed_angle'] = np.where(pitch_data['description'] == 'foul', 0, pitch_data['launch_speed_angle'])
pitch_data['launch_speed_angle'] = np.where(pitch_data['description'].isin(swstr_types), 7, pitch_data['launch_speed_angle'])

# assign binary values for each of the eight swing outcomes, codded by 0-7
pitch_data['swstr']   = np.where(pitch_data['launch_speed_angle'] == 7, 1, 0)
pitch_data['barrel']  = np.where(pitch_data['launch_speed_angle'] == 6, 1, 0)
pitch_data['solid']   = np.where(pitch_data['launch_speed_angle'] == 5, 1, 0)
pitch_data['flr_brn'] = np.where(pitch_data['launch_speed_angle'] == 4, 1, 0)
pitch_data['under']   = np.where(pitch_data['launch_speed_angle'] == 3, 1, 0)
pitch_data['topped']  = np.where(pitch_data['launch_speed_angle'] == 2, 1, 0)
pitch_data['weak']    = np.where(pitch_data['launch_speed_angle'] == 1, 1, 0)
pitch_data['foul']    = np.where(pitch_data['launch_speed_angle'] == 0, 1, 0)



'''
###############################################################################
############################# Classify Pitch Types ############################
###############################################################################
'''

classified_pitch_data = pitch_data.copy()

# function for determining repertoires
def get_repertoire(df, pitcher, year = 'all'):
        
    # number of pitches thrown
    n = len(df)
    if n == 0:
        raise AttributeError('No data for this pitcher & year(s).')
    
    pitch_type_df = df.groupby('pitch_type').agg(count = ('release_speed', 'count'),
                                                 velo = ('release_speed', 'mean'),
                                                 hb = ('pfx_x', 'mean'),
                                                 ivb = ('pfx_z', 'mean'),
                                                 platoon = ('platoon', 'mean')).sort_values(by='count', ascending=False)
    
    # get sinkers and 4-seamers for pitch shape baseline
    try:
        ff_baseline = pitch_type_df.loc['FF']
    except KeyError:
        try:
            si_baseline = pitch_type_df.loc['SI']
            ff_baseline = si_baseline + [0, 1, -5, 8, 0]
        except KeyError:
            try:
                ct_baseline = pitch_type_df.loc['FC']
                if ct_baseline['ivb'] > 6:
                    ff_baseline = ct_baseline + [0, 3, 8, 5, 0]
                else:
                    ff_baseline = ct_baseline + [0, 6, 8, 10, 0]
            except KeyError:
                df['true_pitch_type'] = np.nan
                return df['true_pitch_type']
        
    try:
        si_baseline = pitch_type_df.loc['SI']
    except KeyError:
        si_baseline = ff_baseline + [0, -1, 5, -8, 0]
    
    _, ffvel, ffh, ffv, _ = ff_baseline
    _, sivel, sih, siv, _ = si_baseline
    
    # pitch archetypes
    pitch_archetypes = np.array([
        [ffh, 18, ffvel],           # Riding Fastball
        [ffh, 11, ffvel],           # Fastball
        [sih, siv, sivel],          # Sinker
        # [-2, 13, ffvel-2],          # Cut-Fastball
        [-3, 8, ffvel - 3],         # Cutter
        [-3, 0, ffvel - 9],         # Gyro Slider
        [-8, 0, ffvel - 11],        # Two-Plane Slider
        [-16, 1, ffvel - 14],       # Sweeper
        [-16, -6, ffvel - 15],      # Slurve
        [-8, -12, ffvel - 16],      # Curveball
        [-8, -12, ffvel - 22],      # Slow Curve
        [sih, siv - 2, sivel - 4],  # Movement-Based Changeup
        [sih, siv - 2, sivel - 10], # Velo-Based Changeup
        # [sih, siv - 10, sivel - 8]  # Screwball
    ])
     
    # pitch names
    pitch_names = np.array([
        'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
        'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup', 
        'Velo-Based Changeup', 'Knuckleball'
    ])
    
    for pitch_type, group in pitch_type_df.iterrows():
        pitch_shape = np.array([group['hb'], group['ivb'], group['velo']])

        if pitch_type == 'KN':
            pitch_name = 'Knuckleball'
            df.loc[df['pitch_type'] == pitch_type, 'true_pitch_type'] = pitch_name
            continue

        distances = np.linalg.norm(pitch_archetypes - pitch_shape, axis=1)
        min_index = np.argmin(distances)
        pitch_name = pitch_names[min_index]
        
        if pitch_name in ['Movement-Based Changeup', 'Velo-Based Changeup']:
            if pitch_name == 'Movement-Based Changeup' and sivel - pitch_shape[2] > 6:
                pitch_name = 'Velo-Based Changeup'
            elif pitch_name == 'Velo-Based Changeup' and sivel - pitch_shape[2] <= 6:
                pitch_name = 'Movement-Based Changeup'
        
        df.loc[df['pitch_type'] == pitch_type, 'true_pitch_type'] = pitch_name
        
        # remove pitch from options
        if pitch_name in ['Riding Fastball', 'Fastball']:
            # can't have both
            pitch_archetypes = np.delete(pitch_archetypes, [0, 1], axis=0)
            pitch_names = np.delete(pitch_names, [0, 1], axis=0)
        else:
            pitch_archetypes = np.delete(pitch_archetypes, min_index, axis=0)
            pitch_names = np.delete(pitch_names, min_index, axis=0)
            
    return df['true_pitch_type']

# iterate over pitcher-years and re-classify (should take about one minute)
pitcher_years = classified_pitch_data.groupby(['pitcher', 'game_year']).size().reset_index()[['pitcher', 'game_year']]
for _, row in tqdm(pitcher_years.iterrows()):
    pitcher, year = row['pitcher'], row['game_year']
    mask = (classified_pitch_data['pitcher'] == pitcher) & (classified_pitch_data['game_year'] == year)
    classified_pitch_data.loc[mask, 'true_pitch_type'] = get_repertoire(classified_pitch_data[mask],
                                                                        pitcher=pitcher,
                                                                        year=year)

classified_pitch_data.to_csv('classified_pitch_data.csv')

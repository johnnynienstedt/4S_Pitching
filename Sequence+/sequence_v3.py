#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:34:39 2025

@author: johnnynienstedt
"""

#
# Sequence+
# Johnny Nienstedt 1/22/24
#

# 
# The fourth and final leg of my 4S Pitching model. 
#
# The purpose of this script is to quantify the value that pitchers gain/lose 
# purely from the way their pitches play off each other in sequence. This could
# be due not only to usage patterns, but also pitch shapes and locations. At
# this point I have only considered usage patters, and very roughly at that.
#

# changes from V2:
    # add three micro sequences:
        # match
        # split
        # freeze
    
import numpy as np
import pandas as pd
from tqdm import tqdm



'''
###############################################################################
################################# Import Data #################################
###############################################################################
'''

classified_pitch_data = pd.read_csv('classified_pitch_data.csv')



'''
###############################################################################
############################# Pitch Type Sequence #############################
###############################################################################
'''

# sort pitches sequentially
classified_pitch_data['day'] = [i[-2:] for i in classified_pitch_data['game_date'].values]
classified_pitch_data['month'] = [i[5:7] for i in classified_pitch_data['game_date'].values]

# sort in chronological order
sort_cols = ['game_pk', 'game_year', 'month', 'day', 'at_bat_number', 'pitch_number']
sorted_df = classified_pitch_data.sort_values(by=sort_cols,
                                              ascending=True
                                              ).reset_index(drop=True)

# use same groups as Shape+
pitch_names = np.array([
            'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider',
            'Two-Plane Slider', 'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 
            'Movement-Based Changeup', 'Velo-Based Changeup', 'Knuckleball'])

pitch_classes = {'fastball': ['Riding Fastball', 'Fastball'],
                 'sinker':   ['Sinker'],
                 'breaking': ['Cutter', 'Gyro Slider', 'Two-Plane Slider',
                              'Sweeper', 'Slurve', 'Curveball', 'Slow Curve'],
                 'offspeed': ['Movement-Based Changeup', 'Velo-Based Changeup',
                              'Knuckleball']}

sorted_df['pitch_class'] = np.where(sorted_df['true_pitch_type'].isin(pitch_classes['fastball']), 'fastball', pd.NA)
sorted_df['pitch_class'] = np.where(sorted_df['true_pitch_type'].isin(pitch_classes['sinker']), 'sinker', sorted_df['pitch_class'])
sorted_df['pitch_class'] = np.where(sorted_df['true_pitch_type'].isin(pitch_classes['breaking']), 'breaking', sorted_df['pitch_class'])
sorted_df['pitch_class'] = np.where(sorted_df['true_pitch_type'].isin(pitch_classes['offspeed']), 'offspeed', sorted_df['pitch_class'])

# add qaulities of previous pitch
cols = ['pitch_number', 'at_bat_number', 'pitch_class', 'release_speed', 'plate_x', 'plate_z', 'release_angle_h', 'release_angle_v']
prev_cols = ['prev_n', 'prev_ab', 'prev_pitch', 'prev_velo', 'prev_x', 'prev_z', 'prev_h', 'prev_v']
sorted_df[prev_cols] = pd.NA

locs = sorted_df.loc[:len(sorted_df) - 2, cols]
sorted_df.loc[1:, prev_cols] = locs.values

# remove first pitch of PA
sorted_df.loc[sorted_df['pitch_number'] == 1, prev_cols] = np.nan
sorted_df.loc[sorted_df['at_bat_number'] != sorted_df['prev_ab'], prev_cols] = np.nan

# add two other outcomes of interest
swing_types = ['hit_into_play', 'swinging_strike_blocked', 'swinging_strike', 'foul', 'foul_tip']
sorted_df['called_strike'] = np.where(sorted_df['description'] == 'called_strike', 1, 0)
sorted_df['chase'] = np.where(sorted_df['zone'] > 9, np.where(sorted_df['description'].isin(swing_types), 1, 0), np.nan)

# frequency of each outcome by previous pitch
sequence_matrices = np.empty((10, 4, 4))
outcomes = ['foul', 'weak', 'topped', 'under', 'flr_brn', 'solid', 'barrel', 'swstr', 'called_strike', 'chase']
for i, outcome in tqdm(enumerate(outcomes)):
    for p1, first_pitch in enumerate(pitch_classes.keys()):
        for p2, second_pitch in enumerate(pitch_classes.keys()):
            mask = (sorted_df['prev_pitch'] == first_pitch) & (sorted_df['pitch_class'] == second_pitch)
            sequence_matrices[i, p1, p2] = sorted_df.loc[mask, outcome].mean()


# run value of each outcome
rv = {}
for outcome in outcomes:
    rv[outcome] = sorted_df[sorted_df[outcome] == 1]['delta_run_exp'].mean()
          
sequence_matrix = pd.DataFrame(index=pitch_classes.keys(), columns=pitch_classes.keys())
for p1, pitch_1 in enumerate(pitch_classes.keys()):
    for p2, pitch_2 in enumerate(pitch_classes.keys()):
        sequence_matrix.loc[pitch_1, pitch_2] = sum([-rv[outcome] * sequence_matrices[i,p1,p2] for i, outcome in enumerate(outcomes)])
        
# Convert sequence_matrix to dictionary for faster lookup
sequence_matrix = sequence_matrix.sub(sequence_matrix.mean())
sequence_dict = {
    (idx, col): sequence_matrix.loc[idx, col]
    for idx in sequence_matrix.index
    for col in sequence_matrix.columns
}

# Use vectorized operation with map
sorted_df['sequence_rv'] = list(map(
    lambda x: sequence_dict.get((x[0], x[1]), np.nan),
    zip(sorted_df['prev_pitch'], sorted_df['pitch_class'])
))

sequence_grades_1 = sorted_df[sorted_df['game_year'] > 2022].groupby(['game_year', 'pitcher', 'player_name'])['sequence_rv'].mean().reset_index()



'''
###############################################################################
############################### Micro Sequences ###############################
###############################################################################
'''

# define parameters
RA_small_radius = 0.65
RA_large_radius = 1.45
loc_small_radius = 0.35
loc_large_radius = 1.80

same_release = ((sorted_df['release_angle_h'] - sorted_df['prev_h'])**2 + (sorted_df['release_angle_v'] - sorted_df['prev_v'])**2 < RA_small_radius**2)
diff_release = ((sorted_df['release_angle_h'] - sorted_df['prev_h'])**2 + (sorted_df['release_angle_v'] - sorted_df['prev_v'])**2 > RA_large_radius**2)
same_loc = ((sorted_df['plate_x'] - sorted_df['prev_x'])**2 + (sorted_df['plate_z'] - sorted_df['prev_z'])**2 < loc_small_radius**2)
diff_loc = ((sorted_df['plate_x'] - sorted_df['prev_x'])**2 + (sorted_df['plate_z'] - sorted_df['prev_z'])**2 > loc_large_radius**2)

# our three executable sequences
sorted_df['match'] = np.where((same_release) & (same_loc), 1, 0)
sorted_df['split'] = np.where((same_release) & (diff_loc), 1, 0)
sorted_df['freeze'] = np.where((diff_release) & (same_loc), 1, 0)

sorted_df.loc[sorted_df['freeze'] == 1, 'delta_run_exp'].mean()*1000

sequence_grades_2 = sorted_df.groupby(['game_year', 'pitcher', 'player_name'])[['match', 'split', 'freeze']].mean().reset_index()
sequence_grades = pd.merge(sequence_grades_1, sequence_grades_2, on = ['game_year', 'pitcher', 'player_name'], how='left')

# sequence_grades.to_csv('sequence_grades.csv')

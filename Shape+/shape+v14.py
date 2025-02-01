#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:12:18 2024

@author: johnnynienstedt
"""

#
# Shape+
# Johnny Nienstedt 8/1/24
#

#
# The first leg of my 4S Pitching model.
#
# The purpose of this script is to develop a metric which grades pitch shapes
# with maximum predictive power. This metric will NOT include any information 
# about arm slot, pitch release, or other complimentary pitches, but will 
# implicitly include extension as it relates to effective velocity.
#

# changes from v13:
    # predicting all swing outcomes individually
        # foul, swstr, 6 BIP outcomes
    # removed outcome scale factor
    # changed from induced break to acceleration
    # adjusted approach angles for pitch location
    
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split



'''
###############################################################################
################################# Import Data #################################
###############################################################################
'''

classified_pitch_data = pd.read_csv('classified_pitch_data.csv')



'''
###############################################################################
############################ Create Outcome Models ############################
###############################################################################
'''

# assign run values for each outcome
rv = {'foul': classified_pitch_data[classified_pitch_data['foul'] == 1]['delta_run_exp'].mean(),
      'barrel': classified_pitch_data[classified_pitch_data['barrel'] == 1]['delta_run_exp'].mean(),
      'solid': classified_pitch_data[classified_pitch_data['solid'] == 1]['delta_run_exp'].mean(),
      'flr_brn': classified_pitch_data[classified_pitch_data['flr_brn'] == 1]['delta_run_exp'].mean(),
      'under': classified_pitch_data[classified_pitch_data['under'] == 1]['delta_run_exp'].mean(),
      'topped': classified_pitch_data[classified_pitch_data['topped'] == 1]['delta_run_exp'].mean(),
      'weak': classified_pitch_data[classified_pitch_data['weak'] == 1]['delta_run_exp'].mean(),
      'swstr': classified_pitch_data[classified_pitch_data['swstr'] == 1]['delta_run_exp'].mean()
      }

# columns of interest
x_cols = ['release_speed', 'release_extension', 'ax', 'az', 'loc_adj_VAA', 'loc_adj_HAA']
outcomes = list(rv.keys())
pred_cols = ['predicted_' + x for x in outcomes]
display_cols = ['player_name', 'pitcher', 'game_year', 'true_pitch_type', 'platoon'] + x_cols + ['pfx_x', 'pfx_z'] + outcomes

# pitch names
pitch_names = np.array([
    'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider',
    'Two-Plane Slider', 'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 
    'Movement-Based Changeup', 'Velo-Based Changeup', 'Knuckleball'])

# train in groups
pitch_classes = {'fastball': ['Riding Fastball', 'Fastball'],
                 'sinker':   ['Sinker'],
                 'breaking': ['Cutter', 'Gyro Slider', 'Two-Plane Slider', 'Sweeper', 
                              'Slurve', 'Curveball', 'Slow Curve'],
                 'offspeed': ['Movement-Based Changeup', 'Velo-Based Changeup',
                              'Knuckleball']
                 }

# store models for analysis
models_dict = {'same-handed':     {'fastball': {},
                                   'sinker': {},
                                   'breaking': {},
                                   'offspeed': {}
                                   },
               'opposite-handed': {'fastball': {},
                                   'sinker': {},
                                   'breaking': {},
                                   'offspeed': {}
                                   },
               }

# train on 2021-2022, grade 2023-2024
training_data = classified_pitch_data.loc[classified_pitch_data.game_year < 2023, display_cols]
grading_data = classified_pitch_data.loc[classified_pitch_data.game_year >= 2023, display_cols]

for pitch_class in pitch_classes:
    pitch_types = pitch_classes[pitch_class]
    p_training_data = training_data[training_data.true_pitch_type.isin(pitch_types)]
    p_grading_data = grading_data[grading_data.true_pitch_type.isin(pitch_types)]
    
    print('Grading ' + pitch_class + '...')
    
    for outcome in tqdm(outcomes):
        y_col = outcome
        
        for hand, handedness in enumerate(['same-handed', 'opposite-handed']):
            
            model_data = p_training_data.loc[p_training_data.platoon == hand, x_cols + [y_col]]
            eval_data = p_grading_data.loc[p_grading_data.platoon == hand, x_cols + [y_col]]
            
            Y = model_data.loc[:, y_col].dropna()
            X = model_data.loc[Y.index, x_cols]
            eval_X = eval_data[x_cols]
            
            # split into train/test sets
            xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

            # Calculate the scale_pos_weight based on class imbalance
            n_negative = (ytrain == 0).sum()
            n_positive = (ytrain == 1).sum()
            scale_pos_weight = n_negative / n_positive
            
            
            xgb_model = xgb.XGBClassifier(objective='binary:logistic',
                                          n_estimators = 500,
                                          learning_rate = 0.035,
                                          max_depth = 5,
                                          min_child_weight = 3,
                                          subsample = 0.65,
                                          colsample_bytree = 0.82,
                                          scale_pos_weight=scale_pos_weight,
                                          gamma = 0.4)            
            
            xgb_model.fit(xtrain, ytrain)
            
            calibrated_model = CalibratedClassifierCV(xgb_model, method='sigmoid', cv='prefit')
            calibrated_model.fit(xtrain, ytrain)
                        
            models_dict[handedness][pitch_class][outcome] = calibrated_model
            
            grading_data.loc[eval_X.index, 'predicted_' + outcome] =  calibrated_model.predict_proba(eval_X)[:, 1]
        

# for aesthetic purposes
del x_cols[2:4]
x_cols = x_cols + ['pfx_x', 'pfx_z']

# compute run value via frequency of each outcome
grading_data['shape_rv'] = -sum((rv[x] * grading_data['predicted_' + x] for x in outcomes))
grading_data['shape_rv'] = grading_data['shape_rv'] - grading_data['shape_rv'].mean()
# grading_data.to_csv('Data/graded_shape_data.csv')



'''
###############################################################################
################################ Grade Pitches ################################
###############################################################################
'''

# compute averages for each pitcher's pitch types (split by handedness)
sh_shape_grades = grading_data[grading_data.platoon == 0].groupby(['game_year', 'player_name', 'pitcher', 'true_pitch_type']).mean(numeric_only=True).copy().reset_index()
sh_counts = grading_data.drop(columns=outcomes)[grading_data.platoon == 0].groupby(['game_year', 'player_name', 'pitcher', 'true_pitch_type']).size()
sh_shape_grades.insert(4, 'count', sh_counts.values)
sh_shape_grades[x_cols] = np.round(sh_shape_grades[x_cols], 1)
sh_shape_grades[pred_cols] = np.round(sh_shape_grades[pred_cols], 5)
sh_shape_grades[['pfx_x', 'pfx_z']] = np.round(sh_shape_grades[['pfx_x', 'pfx_z']])

oh_shape_grades = grading_data[grading_data.platoon == 1].groupby(['game_year', 'player_name', 'pitcher', 'true_pitch_type']).mean(numeric_only=True).copy().reset_index()
oh_counts = grading_data.drop(columns=outcomes)[grading_data.platoon == 1].groupby(['game_year', 'player_name', 'pitcher', 'true_pitch_type']).size()
oh_shape_grades.insert(4, 'count', oh_counts.values)
oh_shape_grades[x_cols] = np.round(oh_shape_grades[x_cols], 1)
oh_shape_grades[pred_cols] = np.round(oh_shape_grades[pred_cols], 5)
oh_shape_grades[['pfx_x', 'pfx_z']] = np.round(oh_shape_grades[['pfx_x', 'pfx_z']])

platoon_rate = grading_data.groupby(['game_year', 'pitcher']).mean(numeric_only=True)['platoon']

sh_shape_grades['platoon'] = sh_shape_grades.apply(lambda row: 1 - platoon_rate.loc[(row['game_year'], row['pitcher'])], axis=1)
sh_percentages = grading_data[grading_data.platoon == 0].groupby(['game_year', 'player_name', 'pitcher', 'true_pitch_type']).size() / grading_data[grading_data.platoon == 0].groupby(['game_year', 'pitcher']).size()
sh_shape_grades['percent'] = sh_shape_grades.set_index(['game_year', 'player_name', 'pitcher', 'true_pitch_type']).index.map(sh_percentages)
percent_col = sh_shape_grades.pop('percent')*100
sh_shape_grades.insert(5, 'percent', percent_col)
sh_shape_grades.rename(columns={'true_pitch_type': 'pitch_type', 
                                'zone_value': 'zone_rate',
                                'release_speed': 'Velo',
                                'release_extension': 'Ext',
                                'pfx_x': 'HB',
                                'pfx_z': 'iVB'}, inplace=True)

oh_shape_grades['platoon'] = oh_shape_grades.apply(lambda row: 1 - platoon_rate.loc[(row['game_year'], row['pitcher'])], axis=1)
oh_percentages = grading_data[grading_data.platoon == 1].groupby(['game_year', 'player_name', 'pitcher', 'true_pitch_type']).size() / grading_data[grading_data.platoon == 1].groupby(['game_year', 'pitcher']).size()
oh_shape_grades['percent'] = oh_shape_grades.set_index(['game_year', 'player_name', 'pitcher', 'true_pitch_type']).index.map(oh_percentages)
percent_col = oh_shape_grades.pop('percent')*100
oh_shape_grades.insert(5, 'percent', percent_col)
oh_shape_grades.rename(columns={'true_pitch_type': 'pitch_type', 
                                'zone_value': 'zone_rate',
                                'release_speed': 'Velo',
                                'release_extension': 'Ext',
                                'pfx_x': 'HB',
                                'pfx_z': 'iVB'}, inplace=True)

# normalize to mean 100, stdev 10
all_shape_grades = pd.concat([sh_shape_grades, oh_shape_grades])
cutoff = len(sh_shape_grades)
xRV_mean = all_shape_grades['shape_rv'].mean()
xRV_std = all_shape_grades['shape_rv'].std()
norm_grades = ((all_shape_grades['shape_rv'] - xRV_mean)/xRV_std + 10)*10
all_shape_grades['shape_rv'] = all_shape_grades['shape_rv'] - xRV_mean

all_shape_grades.insert(2, 'Shape+', np.round(norm_grades, 1))
sh_shape_grades.insert(2, 'Shape+', np.round(norm_grades[:cutoff]), 1)
oh_shape_grades.insert(2, 'Shape+', np.round(norm_grades[cutoff:]), 1)

# all_shape_grades.to_csv('Data/pitch_type_shape_grades.csv')

# grade pitch types
sh_pitch_type_grades = sh_shape_grades.groupby('pitch_type').mean(numeric_only=True)['Shape+'].sort_values(ascending=False)
oh_pitch_type_grades = oh_shape_grades.groupby('pitch_type').mean(numeric_only=True)['Shape+'].sort_values(ascending=False)
pitch_type_grades = all_shape_grades.groupby('pitch_type').mean(numeric_only=True)['Shape+'].sort_values(ascending=False)



'''
###############################################################################
############################### Grade Pitchers ################################
###############################################################################
'''

# calculate weights for each row based on pitch counts
all_shape_grades['weight'] = all_shape_grades['count'] / all_shape_grades.groupby(['game_year', 'pitcher'])['count'].transform('sum')

# make weighted columns
all_shape_grades['weighted_Shape+'] = all_shape_grades['Shape+'] * all_shape_grades['weight']
all_shape_grades['weighted_shape_rv'] = all_shape_grades['shape_rv'] * all_shape_grades['weight']
for col in outcomes:
    all_shape_grades['weighted_' + col] = all_shape_grades['predicted_' + col] * all_shape_grades['weight']

weighted_cols = ['weighted_Shape+', 'weighted_shape_rv'] + ['weighted_' + x for x in outcomes]

# sum weights and re-format
shape_grades = all_shape_grades.groupby(['game_year', 'pitcher', 'player_name']).sum([weighted_cols]).reset_index()[['game_year','pitcher', 'player_name'] + weighted_cols].round(5)
shape_grades.rename(columns={col: col.replace('weighted_', '') for col in shape_grades.columns}, inplace=True)
shape_grades[outcomes] = (100*shape_grades[outcomes]).round(1)
shape_grades['count'] = all_shape_grades.groupby(['game_year', 'pitcher']).sum()['count'].values

# re-normalize shape on pitcher level
shape_dist = shape_grades['Shape+'].values
shape_mean = shape_dist.mean()
shape_std = shape_dist.std()
norm_grades = ((shape_dist - shape_mean)/shape_std + 10)*10

shape_grades['Shape+'] = norm_grades.astype(int)

# for aesthetic purposes
def rearrange_name(name):
    last, first = name.split(', ')
    return f"{first} {last}"

shape_grades['Name'] = shape_grades['player_name'].apply(rearrange_name)

# shape_grades.to_csv('Shape+/shape_grades.csv')



'''
###############################################################################
########################## Pitch-Level Correlations ###########################
###############################################################################
'''

shape_corr = all_shape_grades[all_shape_grades['count'] > 100]

def pitcher_pitch_correlations(ind_colname, dep_colname):
    
    x = shape_corr[ind_colname]
    y = shape_corr[dep_colname]
    
    m, b, r, p, std_err = stats.linregress(x, y)
    
    fig, ax = plt.subplots()
    
    xname = ind_colname
    yname = dep_colname

    ax.scatter(x, y, s=3)
    ax.plot(x, m*x+b, '--k')
    ax.set_title(xname + ' vs. ' + yname + ' (Pitcher Pitch Type Level)')
    ax.set_xlabel(xname + ' (min. 100 pitches)')
    ax.set_ylabel(yname)
    
    # ax.set_aspect('equal', adjustable='box')
    
    left = x.min()
    top = y.max()*0.9
    ax.text(left, top, '$R^2$ = ' + str(round(r**2, 2)))
    
    plt.show()
    
# pitcher_pitch_correlations('predicted_foul', 'foul')
# pitcher_pitch_correlations('predicted_barrel', 'barrel')
# pitcher_pitch_correlations('predicted_solid', 'solid')
# pitcher_pitch_correlations('predicted_flr_brn', 'flr_brn')
# pitcher_pitch_correlations('predicted_under', 'under')
# pitcher_pitch_correlations('predicted_topped', 'topped')
# pitcher_pitch_correlations('predicted_weak', 'weak')
# pitcher_pitch_correlations('predicted_swstr', 'swstr')

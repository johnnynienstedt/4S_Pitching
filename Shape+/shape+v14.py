#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:12:18 2024

@author: johnnynienstedt
"""

#
# Early attempts at Shape+
# Johnny Nienstedt 8/1/24
#

#
# The purpose of this script is to develop a metric which grades pitch shapes
# on a standardized scale. This metric will NOT include any information about
# other complimentary pitches, but will include pitcher-specific information
# such as extension and release height.
#
# Ideally, this metric will be used in conjunction with another metric which
# grades the interaction between pitches to grade a pitcher's arsenal. This
# could give insight into which pitchers are getting the most out of their
# pitches by pairing them well and which could be in need of a change to their
# repetiore.
#

# changes from v13:
    # predicting all swing outcomes individually
        # foul, swstr, 6 BIP outcomes
    
    
import pybaseball
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import get_ipython
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



'''
###############################################################################
############################# Import & Clean Data #############################
###############################################################################
'''

# import pitch data from 2021-2024
all_pitch_data = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Data/all_pitch_data', index_col=False)

drop_cols = ['Unnamed: 0', 'level_0', 'index']
necessary_cols = ['release_speed', 'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax',
                  'ay', 'az', 'release_pos_x', 'release_pos_y', 'release_pos_z', 
                  'release_extension']

for col in drop_cols:
    try:
        all_pitch_data.drop(columns = [col], inplace=True)
    except KeyError:
        pass
        
clean_pitch_data = all_pitch_data.dropna(subset = necessary_cols).copy()
clean_pitch_data['pfx_x'] = np.round(clean_pitch_data['pfx_x']*12)
clean_pitch_data['pfx_z'] = np.round(clean_pitch_data['pfx_z']*12)

# select pitchers with at least 100 pitches thrown
pitcher_pitch_data = clean_pitch_data.groupby('pitcher').filter(lambda x: len(x) >= 100)

# flip axis for RHP so that +HB = arm side, -HB = glove side
mirror_cols = ['release_pos_x', 'plate_x', 'pfx_x', 'vx0', 'ax']
pitcher_pitch_data.loc[pitcher_pitch_data['p_throws'] == 'R', mirror_cols] = -pitcher_pitch_data.loc[pitcher_pitch_data['p_throws'] == 'R', mirror_cols]




'''
###############################################################################
######################## Calculate Secondary Parameters #######################
###############################################################################
'''

# get relevant primary parameters
vx0 = pitcher_pitch_data['vx0']
vy0 = pitcher_pitch_data['vy0']
vz0 = pitcher_pitch_data['vz0']
ax = pitcher_pitch_data['ax']
ay = pitcher_pitch_data['ay']
az = pitcher_pitch_data['az']
rx = pitcher_pitch_data['release_pos_x']
ry = pitcher_pitch_data['release_pos_y']
rz = pitcher_pitch_data['release_pos_z']
velo = pitcher_pitch_data['release_speed']
y0 = 50
yf = 17/12

# vertical and horizontal release angle
theta_z0 = -np.arctan(vz0/vy0)*180/np.pi
theta_x0 = -np.arctan(vx0/vy0)*180/np.pi
pitcher_pitch_data['release_angle_v'] = round(theta_z0, 2)
pitcher_pitch_data['release_angle_h'] = round(theta_x0, 2)

# vertical and horizontal approach angle
vyf = -np.sqrt(vy0**2- (2 * ay * (y0 - yf)))
t = (vyf - vy0)/ay
vzf = vz0 + (az*t)
vxf = vx0 + (ax*t)

theta_zf = -np.arctan(vzf/vyf)*180/np.pi
theta_xf = -np.arctan(vxf/vyf)*180/np.pi
pitcher_pitch_data['VAA'] = round(theta_zf, 2)
pitcher_pitch_data['HAA'] = round(theta_xf, 2)

# my calculations of VAA and HAA
zf = pitcher_pitch_data['plate_z']
delta_z = rz - zf
delta_y = ry - yf
phi_z = -np.arctan(delta_z/delta_y)*180/np.pi

xf = pitcher_pitch_data['plate_x']
delta_x = rx - xf
phi_x = -np.arctan(delta_x/delta_y)*180/np.pi

pitcher_pitch_data.insert(89, 'my_VAA', round(2*phi_z - theta_z0, 2))
pitcher_pitch_data.insert(91, 'my_HAA', round(2*phi_x - theta_x0, 2))

delta = []
for v in range(75,102):
    VAA = pitcher_pitch_data[
        (pitcher_pitch_data['release_speed'] > v - 0.5) &
        (pitcher_pitch_data['release_speed'] < v + 0.5)]['VAA'].mean()
    my_VAA = pitcher_pitch_data[
        (pitcher_pitch_data['release_speed'] > v - 0.5) &
        (pitcher_pitch_data['release_speed'] < v + 0.5)
    ]['my_VAA'].mean()
    delta.append(VAA - my_VAA)
    
model = np.poly1d(np.polyfit(np.arange(75,102), delta, 2))

velo_correction = model[0] + model[1]*velo + model[2]*velo**2
geom_VAA = 2*phi_z - theta_z0 + velo_correction
geom_HAA = 2*phi_x - theta_x0 + velo_correction

pitcher_pitch_data.drop(columns=['my_VAA', 'my_HAA'], inplace=True)
pitcher_pitch_data.insert(89, 'geom_VAA', round(geom_VAA, 2))
pitcher_pitch_data.insert(91, 'geom_HAA', round(geom_HAA, 2))

# total break angle
delta_theta_z = theta_z0 - theta_zf
delta_theta_x = theta_x0 - theta_xf
delta_theta = np.sqrt(delta_theta_z**2 + delta_theta_x**2)
pitcher_pitch_data['break_angle'] = round(delta_theta, 2)

# sharpness of break
eff_t = delta_y/velo
iVB = pitcher_pitch_data['pfx_z']
sharpness = np.abs(iVB/eff_t)
pitcher_pitch_data['sharpness'] = round(sharpness, 2)





'''
###############################################################################
############################# Assign Pitch Result #############################
###############################################################################
'''

swstr_types = ['swinging_strike_blocked', 'swinging_strike', 'foul_tip']

# 0 for same handedness, 1 for different
pitcher_pitch_data['platoon'] = np.where(pitcher_pitch_data['p_throws'] == pitcher_pitch_data['stand'], 0, 1) 

# add non-bip swing outcomes
pitcher_pitch_data['launch_speed_angle'] = np.where(pitcher_pitch_data['description'] == 'foul', 0, pitcher_pitch_data['launch_speed_angle'])
pitcher_pitch_data['launch_speed_angle'] = np.where(pitcher_pitch_data['description'].isin(swstr_types), 7, pitcher_pitch_data['launch_speed_angle'])

# assign binary values for each of the eight swing outcomes, codded by 0-7
pitcher_pitch_data['swstr']   = np.where(pitcher_pitch_data['launch_speed_angle'] == 7, 1, 0)
pitcher_pitch_data['barrel']  = np.where(pitcher_pitch_data['launch_speed_angle'] == 6, 1, 0)
pitcher_pitch_data['solid']   = np.where(pitcher_pitch_data['launch_speed_angle'] == 5, 1, 0)
pitcher_pitch_data['flr_brn'] = np.where(pitcher_pitch_data['launch_speed_angle'] == 4, 1, 0)
pitcher_pitch_data['under']   = np.where(pitcher_pitch_data['launch_speed_angle'] == 3, 1, 0)
pitcher_pitch_data['topped']  = np.where(pitcher_pitch_data['launch_speed_angle'] == 2, 1, 0)
pitcher_pitch_data['weak']    = np.where(pitcher_pitch_data['launch_speed_angle'] == 1, 1, 0)
pitcher_pitch_data['foul']    = np.where(pitcher_pitch_data['launch_speed_angle'] == 0, 1, 0)

# assign run values for each outcome
rv = {'foul': pitcher_pitch_data[pitcher_pitch_data['foul'] == 1]['delta_run_exp'].mean(),
      'barrel': pitcher_pitch_data[pitcher_pitch_data['barrel'] == 1]['delta_run_exp'].mean(),
      'solid': pitcher_pitch_data[pitcher_pitch_data['solid'] == 1]['delta_run_exp'].mean(),
      'flr_brn': pitcher_pitch_data[pitcher_pitch_data['flr_brn'] == 1]['delta_run_exp'].mean(),
      'under': pitcher_pitch_data[pitcher_pitch_data['under'] == 1]['delta_run_exp'].mean(),
      'topped': pitcher_pitch_data[pitcher_pitch_data['topped'] == 1]['delta_run_exp'].mean(),
      'weak': pitcher_pitch_data[pitcher_pitch_data['weak'] == 1]['delta_run_exp'].mean(),
      'swstr': pitcher_pitch_data[pitcher_pitch_data['swstr'] == 1]['delta_run_exp'].mean()
      }





'''
###############################################################################
############################# Classify Pitch Types ############################
###############################################################################
'''

classified_pitch_data = pitcher_pitch_data.copy()

# function for determining repertoires
def get_repertoire(pitcher, year = 'all'):
    
    # select proper year(s)
    if year == 'all':
        years = pitcher_pitch_data[pitcher_pitch_data.player_name == pitcher].game_year.unique()
        for year in years:
            get_repertoire(pitcher, year)
        return
    else:
        df = pitcher_pitch_data[(pitcher_pitch_data.player_name == pitcher) & 
                                (pitcher_pitch_data.game_year == year)].copy().reset_index(drop=True)  
    
    # number of pitches thrown
    n = len(df)
    if n == 0:
        raise AttributeError('No data for this pitcher & year(s).')
    
    # percent thrown to same-handed batters
    platoon_percent = (df.stand == df.p_throws).mean() * 100
    classified_pitch_data.loc[(classified_pitch_data.game_year == year) &
                             (classified_pitch_data.player_name == pitcher), 'platoon_percent'] = platoon_percent
    
    
    # get sinkers and 4-seamers for pitch shape baseline
    ff = df[df.pitch_type == 'FF']
    si = df[df.pitch_type == 'SI']
    
    ff_baseline = (ff.release_speed.mean(), ff.pfx_x.mean(), ff.pfx_z.mean())
    si_baseline = (si.release_speed.mean(), si.pfx_x.mean(), si.pfx_z.mean())
    
    ffvel, ffh, ffv = ff_baseline if len(ff) >= 10 else (94, 5, 14)
    sivel, sih, siv = si_baseline if len(si) >= 10 else (93, 12, 9)
    
    # If either pitch type is missing, adjust the baselines accordingly
    if len(si) < 10 and len(ff) > 10:
        sivel, sih, siv = ffvel - 1, ffh + 5, ffv - 5
    if len(ff) < 10 and len(si) > 10:
        ffvel, ffh, ffv = sivel + 1, sih - 5, siv + 5
    
    # pitch archetypes
    pitch_archetypes = np.array([
        [ffh, 18, ffvel],  # Riding Fastball
        [ffh, 10, ffvel],  # Fastball
        [sih, siv, sivel],  # Sinker
        [-3, 8, ffvel - 3],  # Cutter
        [-3, 0, ffvel - 9],  # Gyro Slider
        [-8, 0, ffvel - 11],  # Two-Plane Slider
        [-16, 1, ffvel - 14],  # Sweeper
        [-16, -6, ffvel - 15],  # Slurve
        [-8, -12, ffvel - 16],  # Curveball
        [-8, -12, ffvel - 22], # Slow Curve
        [sih, siv - 5, sivel - 4],  # Movement-Based Changeup
        [sih, siv - 5, sivel - 10]   # Velo-Based Changeup
    ])
     
    # pitch names
    pitch_names = np.array([
        'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
        'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup', 
        'Velo-Based Changeup', 'Knuckleball'
    ])

    
    
    pitch_type_group = df.groupby('pitch_type')
    for pitch_type, group in pitch_type_group:
        pitch_shape = np.array([group.pfx_x.mean(), group.pfx_z.mean(), group.release_speed.mean()])
        if platoon_percent == 0:
            sh_percent = 0
            oh_percent  = 100
        elif platoon_percent == 100:
            sh_percent = 100
            oh_percent  = 0
        else:
            sh_percent = (group.stand == group.p_throws).sum() / round(platoon_percent/100*n) * 100
            oh_percent = (group.stand != group.p_throws).sum() / round((100 - platoon_percent)/100*n) * 100
    
        if pitch_type == 'KN':
            pitch_name = 'Knuckleball'
        else:
            distances = np.linalg.norm(pitch_archetypes - pitch_shape, axis=1)
            min_index = np.argmin(distances)
            pitch_name = pitch_names[min_index]
            
            if pitch_name in ['Movement-Based Changeup', 'Velo-Based Changeup']:
                if pitch_name == 'Movement-Based Changeup' and sivel - pitch_shape[2] > 6:
                    pitch_name = 'Velo-Based Changeup'
                elif pitch_name == 'Velo-Based Changeup' and sivel - pitch_shape[2] <= 6:
                    pitch_name = 'Movement-Based Changeup'
        
        mask = (classified_pitch_data.game_year == year) & \
               (classified_pitch_data.player_name == pitcher) & \
               (classified_pitch_data.pitch_type == pitch_type)
        
        classified_pitch_data.loc[mask, ['true_pitch_type', 'sh_percent', 'oh_percent']] = [
            pitch_name, round(sh_percent, 1), round(oh_percent, 1)
        ]
    
        if pitch_name in ['Riding Fastball', 'Fastball']:
            # Update archetypes and names after identification
            pitch_archetypes = np.delete(pitch_archetypes, min_index, axis=0)
            pitch_names = np.delete(pitch_names, min_index, axis=0)
            

for pitcher in tqdm(pitcher_pitch_data.player_name.unique()):
    get_repertoire(pitcher, year='all')


classified_pitch_data.to_csv('classified_pitch_data.csv')





'''
###############################################################################
############################ Create Outcome Models ############################
###############################################################################
'''

# columns of interest
x_cols = ['release_speed', 'release_extension', 'pfx_x', 'pfx_z', 'VAA', 'HAA',]
outcomes = list(rv.keys())
pred_cols = ['predicted_' + x for x in outcomes]
display_cols = ['player_name', 'pitcher', 'true_pitch_type', 'platoon'] + x_cols + outcomes

# pitch names
pitch_names = np.array([
    'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
    'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup', 
    'Velo-Based Changeup', 'Knuckleball'
])

pitch_classes = {'fastball': ['Riding Fastball', 'Fastball', 'Sinker'],
                 'breaking': ['Cutter', 'Gyro Slider', 'Two-Plane Slider', 'Sweeper', 'Slurve', 'Curveball', 'Slow Curve'],
                 'offspeed': ['Movement-Based Changeup', 'Velo-Based Changeup', 'Knuckleball']
                 }

models_dict = {'same-handed': {'fastball': {},
                               'breaking': {},
                               'offspeed': {}
                               },
               'opposite-handed': {'fastball': {},
                                   'breaking': {},
                                   'offspeed': {}
                                   }
               }

training_data = classified_pitch_data.loc[classified_pitch_data.game_year < 2024, display_cols]
grading_data = classified_pitch_data.loc[classified_pitch_data.game_year == 2024, display_cols]

for pitch_class in pitch_classes:
    pitch_types = pitch_classes[pitch_class]
    p_training_data = training_data[training_data.true_pitch_type.isin(pitch_types)]
    p_grading_data = grading_data[grading_data.true_pitch_type.isin(pitch_types)]
    
    for outcome in outcomes:
        y_col = outcome
        
        for handedness in enumerate(['same-handed', 'opposite-handed']):
            
            model_data = p_training_data.loc[p_training_data.platoon == handedness[0], x_cols + [y_col]]
            eval_data = p_grading_data.loc[p_grading_data.platoon == handedness[0], x_cols + [y_col]]
            
            Y = model_data.loc[:, y_col].dropna()
            X = model_data.loc[Y.index, x_cols]
            eval_X = eval_data[x_cols]
                    
            # split into train/test sets
            xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
            
            xgb_model = xgb.XGBClassifier(objective='binary:logistic',
                                          n_estimators = 500,
                                          learning_rate = 0.035,
                                          max_depth = 5,
                                          min_child_weight = 2,
                                          subsample = 0.65,
                                          colsample_bytree = 0.82,
                                          gamma = 0.4)
    
            xgb_model.fit(xtrain, ytrain)
            
            models_dict[handedness[1]][pitch_class][outcome] = xgb_model
            
            grading_data.loc[eval_X.index, 'predicted_' + outcome] = xgb_model.predict_proba(eval_X)[:, 1]
        

scale_factor = np.sum(grading_data[pred_cols], axis=1)

grading_data[pred_cols] = grading_data[pred_cols].div(scale_factor, axis=0)

# plt.figure
# plt.bar(x_cols, models_dict['same-handed']['fastball']['flr_brn'].feature_importances_)
# plt.show()


'''
###############################################################################
################################ Grade Pitches ################################
###############################################################################
'''

# compute averages for each pitcher's pitch types
sh_shape_grades = grading_data[grading_data.platoon == 0].groupby(['player_name', 'pitcher', 'true_pitch_type']).mean(numeric_only=True).copy().reset_index()
sh_counts = grading_data.drop(columns=outcomes)[grading_data.platoon == 0].groupby(['player_name', 'pitcher', 'true_pitch_type']).size()
sh_shape_grades.insert(4, 'count', sh_counts.values)
sh_shape_grades[x_cols] = np.round(sh_shape_grades[x_cols], 1)
sh_shape_grades[pred_cols] = np.round(sh_shape_grades[pred_cols], 3)
sh_shape_grades[['pfx_x', 'pfx_z']] = np.round(sh_shape_grades[['pfx_x', 'pfx_z']])

oh_shape_grades = grading_data[grading_data.platoon == 1].groupby(['player_name', 'pitcher', 'true_pitch_type']).mean(numeric_only=True).copy().reset_index()
oh_counts = grading_data.drop(columns=outcomes)[grading_data.platoon == 1].groupby(['player_name', 'pitcher', 'true_pitch_type']).size()
oh_shape_grades.insert(4, 'count', oh_counts.values)
oh_shape_grades[x_cols] = np.round(oh_shape_grades[x_cols], 1)
oh_shape_grades[pred_cols] = np.round(oh_shape_grades[pred_cols], 3)
oh_shape_grades[['pfx_x', 'pfx_z']] = np.round(oh_shape_grades[['pfx_x', 'pfx_z']])

platoon_rate = grading_data.groupby('pitcher').mean(numeric_only=True)['platoon']

sh_shape_grades['platoon'] = sh_shape_grades['pitcher'].map(1 - platoon_rate)
sh_percentages = grading_data[grading_data.platoon == 0].groupby(['player_name', 'pitcher', 'true_pitch_type']).size() / grading_data[grading_data.platoon == 0].groupby(['pitcher']).size()
sh_shape_grades['percent'] = sh_shape_grades.set_index(['player_name', 'pitcher', 'true_pitch_type']).index.map(sh_percentages)
percent_col = sh_shape_grades.pop('percent')*100
sh_shape_grades.insert(5, 'percent', percent_col)
sh_shape_grades.rename(columns={'true_pitch_type': 'pitch_type', 
                                'zone_value': 'zone_rate',
                                'release_speed': 'Velo',
                                'release_extension': 'Ext',
                                'pfx_x': 'HB',
                                'pfx_z': 'iVB'}, inplace=True)

oh_shape_grades['platoon'] = oh_shape_grades['pitcher'].map(platoon_rate)
oh_percentages = grading_data[grading_data.platoon == 1].groupby(['player_name', 'pitcher', 'true_pitch_type']).size() / grading_data[grading_data.platoon == 1].groupby(['pitcher']).size()
oh_shape_grades['percent'] = oh_shape_grades.set_index(['player_name', 'pitcher', 'true_pitch_type']).index.map(oh_percentages)
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

xRV = sum((rv[x] * all_shape_grades['predicted_' + x] for x in outcomes))
xRV_mean = xRV.mean()
xRV_std = xRV.std()
norm_grades = ((xRV_mean-xRV)/xRV_std + 10)*10

all_shape_grades.insert(2, 'Shape+', np.round(norm_grades, 1))
sh_shape_grades.insert(2, 'Shape+', np.round(norm_grades[:cutoff]), 1)
oh_shape_grades.insert(2, 'Shape+', np.round(norm_grades[cutoff:]), 1)



#
# Grade pitch types
#

sh_pitch_type_grades = sh_shape_grades.groupby('pitch_type').mean(numeric_only=True)['Shape+']
oh_pitch_type_grades = oh_shape_grades.groupby('pitch_type').mean(numeric_only=True)['Shape+']
pitch_type_grades = all_shape_grades.groupby('pitch_type').mean(numeric_only=True)['Shape+']

# def grade_pitch(pitch_class, shape):
    
#     names = x_cols
#     shape_df = pd.DataFrame([shape], columns = names)
    
#     probabilities = pd.DataFrame()
#     for outcome in outcomes:
        
#         sh_model = models_dict['same-handed'][pitch_class][outcome]
#         oh_model = models_dict['opposite-handed'][pitch_class][outcome]
        
#         probabilities.loc[0, outcome] = sh_model.predict(shape_df)[0]
#         probabilities.loc[1, outcome] = oh_model.predict(shape_df)[0]
    
#     xRV = probabilities.whiff*swstr_rv + probabilities.gb*gb_rv + probabilities.fb*fb_rv
    
#     grades = ((xRV - xRV_mean)/xRV_std + 10)*10
    
#     print()
#     print(shape_df.to_string(index=False))
#     print()
#     print('    Same hand Shape+:', round(grades[0]))
#     print('Opposite hand Shape+:', round(grades[1]))




'''
###############################################################################
############################### Grade Pitchers ################################
###############################################################################
'''

# display repertoire and grades
def grade_repertoire(pitcher, verbose = True, backend = 'qt'):
    
    # pitcher first and last name
    pfirst = pitcher.split(', ')[1]
    plast = pitcher.split(', ')[0]
    
    n = len(all_shape_grades[all_shape_grades.player_name == pitcher]['pitcher'].unique())
    
    if n == 0:
        print('\nNo data for ' + pfirst + ' ' + plast + ' in 2024. Check for spelling or missing accents/apostrophes.')
        return
    elif n > 1:
        print('\nThere are two pitchers with this name. Unfortunately this problem cannot be resolved at this time.')
        return
    
    # get all pitches from this year
    df = classified_pitch_data.copy().query('player_name == @pitcher and game_year == 2024', engine='python')
    
    # pitcher handedness
    hand = df.p_throws.values[0]
    
    # get grades + shapes
    if hand == 'R':
        rhb = sh_shape_grades.query('player_name == @pitcher')
        lhb = oh_shape_grades.query('player_name == @pitcher')
    else:
        rhb = oh_shape_grades.query('player_name == @pitcher')
        lhb = sh_shape_grades.query('player_name == @pitcher')



    #
    # plot all pitches by type
    #
    
    if verbose:
        
        # select plotting method
        if backend == 'qt':
            ipython = get_ipython()
            ipython.run_line_magic('matplotlib', 'qt')
        else:
            ipython = get_ipython()
            ipython.run_line_magic('matplotlib', 'inline')
        
        # pitches + corresponding colors
        pitch_names = np.array([
            'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
            'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup', 
            'Velo-Based Changeup', 'Knuckleball'
        ])
        
        colors = np.array(['red', 'tomato', 'darkorange', 'sienna', 'xkcd:sunflower yellow', 
                           'yellowgreen', 'limegreen', 'lightseagreen', 
                           'cornflowerblue', 'mediumpurple', 'darkgoldenrod',
                           'goldenrod', 'gray'])
        
        colordict = dict(zip(pitch_names, colors))
    
        # add colors
        df['color'] = df['true_pitch_type'].map(colordict)
        df = df.dropna(subset = 'color')
    
        # redo this for plotting
        pitch_names = np.array([
            'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
            'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based\n Changeup', 
            'Velo-Based\n Changeup', 'Knuckleball'
        ])
        
        colordict = dict(zip(pitch_names, colors))
    
        # sort by usage
        df = df.sort_values(by='sh_percent', ascending=False)
        
        # all pitch shapes
        HB = list(df.pfx_x)
        iVB = list(df.pfx_z)
        velo = list(df.release_speed)
        
        # make plot
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0.05)
        ax.scatter(velo, HB, iVB, c=df.color)
        
        # make legend handles and labels
        pitch_arsenal = list(df.true_pitch_type.unique())
        try:
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colordict[pitch], markersize=10) for pitch in pitch_arsenal]
        except KeyError:
            for i in range(len(pitch_arsenal)):
                if pitch_arsenal[i] in ('Movement-Based Changeup', 'Velo-Based Changeup'):
                    pitch_arsenal[i] = pitch_arsenal[i].split()[0] + '\n ' + pitch_arsenal[i].split()[1]
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colordict[pitch], markersize=10) for pitch in pitch_arsenal]
        
        # sort handles and labels
        legend_labels = pitch_arsenal
        sorted_handles_labels = [(handles[i], legend_labels[i]) for i in range(len(legend_labels))]
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        
        # make legend
        legend = ax.legend(sorted_handles, sorted_labels, loc='center left', bbox_to_anchor=(-0.2, 0.6), fontsize = 15)
        ax.add_artist(legend)
        
        # set title
        fig.suptitle('2024 Pitch Repertoire -- ' + pfirst + ' ' + plast, fontsize=20)
        ax.set_xlabel('Velo', fontsize=15)
        ax.set_ylabel('HB', fontsize=15)
        ax.set_zlabel('iVB', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim((100,70))

        if hand == 'R': 
            ax.set_ylim((-20,20))
        else: 
            ax.set_ylim((20,-20)) 

        ax.set_zlim((-20,20))
        
        # make home plate
        vertices = np.array([
            [102, -8.5, -29],  # Front left
            [102, 8.5, -29],   # Front right
            [110.5, 8.5, -29], # Back right
            [119, 0, -29],     # Back point
            [110.5, -8.5, -29] # Back left
        ])
        
        faces = [[vertices[i] for i in range(5)]]
        poly = Poly3DCollection(faces, alpha=0.8, facecolor='white', edgecolor='black')
        ax.add_collection3d(poly)
        
        # # make strike zone
        # vertices = np.array([
        #     [102, -8.5, -11],  # Top left
        #     [102, 8.5, -11],   # Top right
        #     [102, 8.5, 10.5],   # Bottom right
        #     [102, -8.5, 10.5],  # Bottom left

        # ])
        
        # faces = [[vertices[i] for i in range(4)]]
        # poly = Poly3DCollection(faces, alpha=0.4, facecolor='white', edgecolor='black')
        # ax.add_collection3d(poly)
        
        # set viewing angle
        ax.view_init(elev=20, azim=-45)

        plt.show()
        
        
        
    #
    # print repertoire grades
    #
    
    # repertoire grades
    rhb_repertoire = rhb[['pitch_type', 'percent', 'Shape+']].reset_index(drop=True)
    lhb_repertoire = lhb[['pitch_type', 'percent', 'Shape+']].reset_index(drop=True)
    
    repertoire = pd.merge(rhb_repertoire, lhb_repertoire, how='outer', on='pitch_type', suffixes=('_R', '_L'))

    repertoire.rename(columns={'percent_R': 'pct_R',
                               'percent_L': 'pct_L'}, inplace=True)
        
    shapes = df[df.player_name == pitcher].groupby(['true_pitch_type']).mean(numeric_only=True)[['release_speed', 'pfx_x', 'pfx_z']].reset_index()
    shapes.rename(columns={'true_pitch_type': 'pitch_type', 
                           'release_speed': 'Velo',
                           'pfx_x': 'HB',
                           'pfx_z': 'iVB'}, inplace=True)
    
    repertoire = pd.merge(shapes, repertoire, on='pitch_type')
    
    # fill NAs for unused pitches
    repertoire = repertoire.fillna(0)
    
    repertoire[['HB', 'iVB', 'Shape+_R', 'Shape+_L']] = repertoire[['HB', 'iVB', 'Shape+_R', 'Shape+_L']].astype(int)
    repertoire[['Velo', 'pct_R', 'pct_L']] = round(repertoire[['Velo', 'pct_R', 'pct_L']], 1)
    
    # sort by more prominent usage
    if hand == 'R':
        RHB_percent = round(df.platoon_percent.values[0], 1)
    else:
        RHB_percent = round(100 - df.platoon_percent.values[0], 1)
    
    if RHB_percent > 0.5:
        repertoire = repertoire.sort_values(by = 'pct_R', ascending=False).reset_index(drop=True)
    else:
        repertoire = repertoire.sort_values(by = 'pct_L', ascending=False).reset_index(drop=True)
    
    # add totals row
    rhb_shape = (rhb['Shape+'] * rhb['count']).sum() / rhb['count'].sum()
    lhb_shape = (lhb['Shape+'] * lhb['count']).sum() / lhb['count'].sum()
    total_shape = (rhb_shape*rhb['count'].sum() + lhb_shape*lhb['count'].sum()) / (rhb['count'].sum() + lhb['count'].sum())
    total_row = ['Total', '', '', '', RHB_percent, rhb_shape.astype(int), 100 - RHB_percent, lhb_shape.astype(int)]
    repertoire.loc[len(repertoire)] =  ['']*repertoire.shape[1]
    repertoire.loc[len(repertoire)] =  total_row
    
    # for readability 
    repertoire = repertoire.rename(columns={'pitch_type': 'Pitch Type'})   
    repertoire['Shape+_R'] = repertoire['Shape+_R'].replace({0: '--'})
    repertoire['Shape+_L'] = repertoire['Shape+_L'].replace({0: '--'})

    print()
    print(pfirst, plast, '(' + hand + 'HP) - 2024')
    print()
    print(repertoire.to_string(index=False))
    print()
    print('Total Shape+:', total_shape.astype(int))
    
    
grade_repertoire('Woo, Bryan', verbose=True)



#
# Calculate aggregate grades for pitchers
#

# calculate weights for each row based on pitch counts
all_shape_grades['weight'] = all_shape_grades['count'] / all_shape_grades.groupby('pitcher')['count'].transform('sum')

# make weighted columns
all_shape_grades['weighted_Shape+'] = all_shape_grades['Shape+'] * all_shape_grades['weight']
for col in outcomes:
    all_shape_grades['weighted_' + col] = all_shape_grades['predicted_' + col] * all_shape_grades['weight']

weighted_cols = ['weighted_Shape+'] + ['weighted_' + x for x in outcomes]

# sum weights and re-format
pitcher_grades = all_shape_grades.groupby(['pitcher', 'player_name']).sum([weighted_cols]).reset_index()[['pitcher', 'player_name'] + weighted_cols].round(3)
pitcher_grades.rename(columns={col: col.replace('weighted_', '') for col in pitcher_grades.columns}, inplace=True)

# re--normalize shape on pitcher level
shape_dist = pitcher_grades['Shape+'].values
shape_mean = shape_dist.mean()
shape_std = shape_dist.std()
norm_grades = ((shape_dist - shape_mean)/shape_std + 10)*10

pitcher_grades['Shape+'] = norm_grades.astype(int)

def rearrange_name(name):
    last, first = name.split(', ')
    return f"{first} {last}"

pitcher_grades['Name'] = pitcher_grades['player_name'].apply(rearrange_name)

pitcher_grades.to_csv('shape_grades.csv')

'''
###############################################################################
########################## Pitch-Level Correlations ###########################
###############################################################################
'''
    
sh_shape_corr = sh_shape_grades[sh_shape_grades['count'] > 100]
oh_shape_corr = oh_shape_grades[oh_shape_grades['count'] > 100]


def pitcher_pitch_correlations(ind_colname, dep_colname, h='s'):
    
    ipython = get_ipython()
    ipython.run_line_magic('matplotlib', 'inline')
    
    if h == 's':
        x = sh_shape_corr[ind_colname]
        y = sh_shape_corr[dep_colname]
    
    else:
        x = oh_shape_corr[ind_colname]
        y = oh_shape_corr[dep_colname]
    
    m, b, r, p, std_err = stats.linregress(x, y)
    
    fig, ax = plt.subplots()
    
    xname = ind_colname
    yname = dep_colname

    ax.scatter(x, y, s=3)
    ax.plot(x, m*x+b, '--k')
    ax.set_title(xname + ' vs. ' + yname)
    ax.set_xlabel(xname + ' (Pitcher Pitch Type Level)')
    ax.set_ylabel(yname)
    
    # ax.set_aspect('equal', adjustable='box')
    
    left = x.min()
    top = y.max()*0.9
    ax.text(left, top, '$R^2$ = ' + str(round(r**2, 2)))
    
    plt.show()
    
    
pitcher_pitch_correlations('predicted_foul', 'foul')
pitcher_pitch_correlations('predicted_barrel', 'barrel')
pitcher_pitch_correlations('predicted_solid', 'solid')
pitcher_pitch_correlations('predicted_flr_brn', 'flr_brn')
pitcher_pitch_correlations('predicted_under', 'under')
pitcher_pitch_correlations('predicted_topped', 'topped')
pitcher_pitch_correlations('predicted_weak', 'weak')
pitcher_pitch_correlations('predicted_swstr', 'swstr')

    
    
    
'''
###############################################################################
######################### Pitcher-Level Correlations ##########################
###############################################################################
'''
    
pitcher_results_24 = pybaseball.pitching_stats(2024, qual = 1, ind = 1)

# pitcher_results.insert(1, 'pitcher', pybaseball.playerid_reverse_lookup(pitcher_results.IDfg, key_type='fangraphs').key_mlbam)
# pitcher_results['pitcher'] = pitcher_results.groupby('Name')['pitcher'].transform(lambda x: x.ffill().bfill())

pitcher_results_24 = pd.merge(pitcher_grades, pitcher_results_24, on='Name', how='inner')


def pitcher_correlations(ind_colname, dep_colname, q=400):
        
    filtered_results = pitcher_results_24[pitcher_results_24.Pitches > q]
    
    ipython = get_ipython()
    ipython.run_line_magic('matplotlib', 'inline')
    
    x = filtered_results[ind_colname]
    y = filtered_results[dep_colname]

    m, b, r, p, std_err = stats.linregress(x, y)
    
    fig, ax = plt.subplots()

    ax.scatter(x, y, s=3)
    ax.plot(x, m*x+b, '--k')
    ax.set_title('2024 ' + ind_colname + ' vs. ' + dep_colname + ' (min. ' + str(q) + ' pitches)')
    ax.set_xlabel(ind_colname + ' (Pitcher Level)')
    ax.set_ylabel(dep_colname)
    
    # ax.set_aspect('equal', adjustable='box')
    
    left = x.min()
    top = y.max()*0.9
    ax.text(left, top, '$R^2$ = ' + str(round(r**2, 2)))
    
    plt.show()

    
pitcher_correlations('swstr', 'SwStr%', q=400)
pitcher_correlations('topped', 'GB%', q=400)
pitcher_correlations('under', 'FB%', q=400)





'''
###############################################################################
########################## Pitcher-Level Stickiness ###########################
###############################################################################
'''

pitcher_results = pybaseball.pitching_stats(2021, 2024, qual = 1, ind = 1)
# pitcher_results.sort_values(by = ['IDfg', 'Season'], inplace=True)
# pitcher_results['Whiff%'] = pitcher_results['whiff%']/pitcher_results['Swing%']
# pitcher_results['Whiff% (sc)'] = pitcher_results['whiff%']/pitcher_results['Swing% (sc)']


# Stickiness year over year
def yoy(df, colname, q=400):
    
    filtered_results = df[df.Pitches > q].reset_index()
    filtered_results.dropna(subset = [colname], inplace=True)
    filtered_results.sort_values(by='IDfg', inplace=True)
    filtered_results.reset_index(drop=True, inplace=True)

    rows = []
    for i in range(len(filtered_results) - 1):
        if filtered_results['IDfg'][i + 1] == filtered_results['IDfg'][i]:
            c1 = filtered_results[colname][i]
            c2 = filtered_results[colname][i + 1]
            c3 = filtered_results['Pitches'][i]
            rows.append({'Year1': c1,
                          'Year2': c2,
                          'N_P': c3})
            
            
    pairs = pd.DataFrame(rows)
    
    
    # make plot
    x = pairs.Year1
    y = pairs.Year2
    
    m, b, r, p, std_err = stats.linregress(x, y)
    
    fig, ax = plt.subplots()
    plt.scatter(x, y, s=3)
    plt.plot(x, m*x+b, '--k')
    ax.set_title(colname + " YOY (Pitcher Level, min. " + str(q) + " Pitches Per Season)")
    ax.set_xlabel('Year 1 ' + colname)
    ax.set_ylabel('Year 2 ' + colname)
        
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

    return pairs

pairs = yoy(pitcher_results, 'Pitching+', q=1000)



# # next steps:

# # def compare_pitchers(pitcher_list):
# # run values from each outcome pls












'''
###############################################################################
########################### Repertoire Colinearity ############################
###############################################################################
'''

# lin_df = pd.DataFrame(columns = ['Name', 'horizontal', 'vertical', 'movement'], index = classified_pitch_data.pitcher.unique(), dtype=float)

# for pitcher in tqdm(classified_pitch_data.pitcher.unique()):
    
#     df = classified_pitch_data.query('pitcher == @pitcher and game_year == 2024').copy()
#     if len(df) < 100:
#         continue
    
#     name = df['player_name'].values[0]
    
#     new_name = name.split(', ')[1] + ' ' + name.split(',')[0]
    
#     lin_df.loc[pitcher, 'Name'] = new_name
    
#     velo = df['release_speed']
#     hb = df['pfx_x']
#     ivb = df['pfx_z']
    
#     # horizontal
#     m, b, r, p, std_err = stats.linregress(velo, hb)
#     lin_df.loc[pitcher, 'horizontal'] = r**2
    
#     # vertical
#     m, b, r, p, std_err = stats.linregress(velo, ivb)
#     lin_df.loc[pitcher, 'vertical'] = r**2
    
#     # movement
#     m, b, r, p, std_err = stats.linregress(hb, ivb)
#     lin_df.loc[pitcher, 'movement'] = r**2
        
# lin_df.dropna(inplace=True)
# lin_df['linearity'] = np.sqrt(lin_df['horizontal']**2 + lin_df['vertical']**2 + lin_df['movement']**2)
# pitcher_results = pybaseball.pitching_stats(2024, qual = 20)
# swstr = pitcher_results[['Name', 'SwStr%', 'SIERA']]
# rep_df = pd.merge(lin_df, swstr, on='Name', how='left').dropna()


# x = rep_df['horizontal']
# y = rep_df['SIERA']

# m, b, r, p, std_err = stats.linregress(x, y)

# if p < 0.05:

    
#     fig, ax = plt.subplots()
#     plt.scatter(x, y)
#     plt.plot([x.min(), x.max()],
#              [x.min()*m + b, x.max()*m + b],
#              '--', color = 'gray')
    
#     x_limits = ax.get_xlim()
#     y_limits = ax.get_ylim()
    
#     y_pos = y_limits[0] + (y_limits[1] - y_limits[0]) * 0.9
    
#     if m > 0:
#         x_pos = x_limits[0] + (x_limits[1] - x_limits[0]) * 0.1
#         ax.text(x_pos, y_pos, f'$R^2$ = {round(r**2, 2)}', ha='left', va='top', fontsize = 12)
#     else:
#         x_pos = x_limits[0] + (x_limits[1] - x_limits[0]) * 0.9
#         ax.text(x_pos, y_pos, f'$R^2$ = {round(r**2, 2)}', ha='right', va='top', fontsize = 12)
#     plt.show()
    
# else: print(p)
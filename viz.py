#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:10:07 2025

@author: johnnynienstedt
"""

#
# Pitcher Grade Visualizations for 4S+
# Johnny Nienstedt 1/20/24
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from PIL import ImageFont
from IPython import get_ipython
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



'''
###############################################################################
################################# Import Data #################################
###############################################################################
'''

pitcher_grades = pd.read_csv('pitcher_grades.csv')
pitch_type_shape_grades = pd.read_csv('pitch_type_shape_grades.csv')
classified_pitch_data = pd.read_csv('classified_pitch_data.csv')



'''
###############################################################################
############################# Function Definition #############################
###############################################################################
'''

def get_data(classified_pitch_data, pitcher_grades, name, year):
    
    df = pitcher_grades[(pitcher_grades['Name'] == name) & (pitcher_grades['game_year'] == year)]
    if len(df) == 0:
        print(f'\nNo data for {name} in {year}. Check spelling?')
        return ['fail', 'fail', 'fail', 'fail']
        
    pitcher_data = {
        "Name": name,
        "Year": int(df['game_year'].values[0]),
        "SP": int(df['starter'].values[0]),
        "MLBAM_ID": df['pitcher'].values[0],
        "Pitches Thrown": df['Pitches'].values[0],
        "4S+": int(df['4S+'].values[0]),
    }
    
    hand = classified_pitch_data.loc[classified_pitch_data['pitcher'] == pitcher_data['MLBAM_ID']]['p_throws'].values[0]
    pitcher_data['Hand'] = hand
    
    shape = df['Shape+'].values[0]
    spot = df['Spot+'].values[0]
    slot = df['Slot+'].values[0]
    sequence = df['Sequence+'].values[0]

    shape_rv = 0.42 * (shape - 100)/10 * (5/6) / 100
    spot_rv = 0.84 * (spot - 100)/10 * (5/6) / 100
    slot_rv = 0.06 * (slot - 100)/10 * (5/6) / 100
    sequence_rv = 0.05 * (sequence - 100)/10 * (5/6) /100
    total_rv = shape_rv + spot_rv + slot_rv + sequence_rv
    
    # rvs = np.array([shape_rv*73, slot_rv*0, sequence_rv*0, spot_rv*27])
    
    # scaling_factor = total_rv / sum(rvs)
    
    # adj_shape_rv = shape_rv*73 * scaling_factor
    # adj_spot_rv = spot_rv*27 * scaling_factor
    # adj_slot_rv = slot_rv*0 * scaling_factor
    # adj_sequence_rv = sequence_rv*0 * scaling_factor
        
    grades = {
        "Shape": int(shape),
        "Slot": int(slot),
        "Sequence": int(sequence),
        "Spot": int(spot),
    }
    
    xrv = {
        "Shape_RV": shape_rv,
        "Slot_RV": slot_rv,
        "Sequence_RV": sequence_rv,
        "Spot_RV": spot_rv,
        "xRV": total_rv
    }
    
    performance_data = pd.DataFrame({
        "": ["Actual", "Deserved", "Predicted"],
        "K%": [f'{df["K%"].values[0]:.1f}', f'{df["deserved_K%"].values[0]:.1f}', f'{df["predicted_K%"].values[0]:.1f}'],
        "BB%": [f'{df["BB%"].values[0]:.1f}', f'{df["deserved_BB%"].values[0]:.1f}', f'{df["predicted_BB%"].values[0]:.1f}'],
        "SIERA": [f'{df["SIERA"].values[0]:.2f}', f'{df["deserved_SIERA"].values[0]:.2f}', f'{df["predicted_SIERA"].values[0]:.2f}'],
    })
    
    return pitcher_data, grades, xrv, performance_data

def grade_repertoire(pitcher, year, pitch_type_shape_grades, classified_pitch_data, display_mode='Shape'):
    
    # get all pitches from this year
    df = classified_pitch_data.query('pitcher == @pitcher and game_year == @year').copy()
    
    # pitcher handedness
    hand = df.p_throws.values[0]
    
    for i in range(1, len(pitch_type_shape_grades['Unnamed: 0'])):
            if pitch_type_shape_grades['Unnamed: 0'][i] < pitch_type_shape_grades['Unnamed: 0'][i - 1]:
                cutoff = i
    
    sh_shape_grades = pitch_type_shape_grades[:cutoff]
    oh_shape_grades = pitch_type_shape_grades[cutoff:]
    
    # get grades + shapes
    if hand == 'R':
        rhb = sh_shape_grades.query('pitcher == @pitcher and game_year == @year')
        lhb = oh_shape_grades.query('pitcher == @pitcher and game_year == @year')
    else:
        rhb = oh_shape_grades.query('pitcher == @pitcher and game_year == @year')
        lhb = sh_shape_grades.query('pitcher == @pitcher and game_year == @year')
        
    #
    # get repertoire grades
    #
    
    # repertoire grades
    rhb_repertoire = rhb[['pitch_type', 'percent', 'Shape+']].reset_index(drop=True)
    lhb_repertoire = lhb[['pitch_type', 'percent', 'Shape+']].reset_index(drop=True)
    
    repertoire = pd.merge(rhb_repertoire, lhb_repertoire, how='outer', on='pitch_type', suffixes=('_R', '_L'))

    repertoire.rename(columns={'percent_R': 'R',
                                'percent_L': 'L'}, inplace=True)
        
    shapes = df.groupby(['true_pitch_type']).mean(numeric_only=True)[['release_speed', 'pfx_x', 'pfx_z']].reset_index()
    shapes.rename(columns={'true_pitch_type': 'pitch_type', 
                            'release_speed': 'Velo',
                            'pfx_x': 'HB',
                            'pfx_z': 'iVB'}, inplace=True)
    
    repertoire = pd.merge(shapes, repertoire, on='pitch_type')

    # fill NAs for unused pitches
    repertoire = repertoire.fillna(0)
    try:
        repertoire.drop(repertoire[repertoire['R'] + repertoire['L'] < 1].index[0], inplace=True)
    except IndexError:
        pass
        
    pitch_type_grades = pitch_type_shape_grades.groupby('pitch_type').mean(numeric_only=True)['Shape+']
    shapes = (repertoire['Shape+_R']*repertoire['R'] + repertoire['Shape+_L']*repertoire['L'])/(repertoire['R'] + repertoire['L'])
    
    repertoire['default'] = pitch_type_grades[repertoire['pitch_type']].values
    raw_grade = shapes - repertoire['default'] + 50
    repertoire['Sct. Grade'] = [int(5 * round(float(i)/5)) for i in raw_grade.values]
    repertoire.drop(columns='default', inplace=True)
    
    # round
    repertoire[['HB', 'iVB', 'Shape+_R', 'Shape+_L', 'R', 'L']] = repertoire[['HB', 'iVB', 'Shape+_R', 'Shape+_L', 'R', 'L']].astype(int)
    repertoire['Velo'] = repertoire['Velo'].round(1)
    
    # sort by more prominent usage
    if hand == 'R':
        RHB_percent = round(df['platoon'].values[0], 1)
    else:
        RHB_percent = round(100 - df['platoon'].values[0], 1)
    
    if RHB_percent > 0.5:
        repertoire = repertoire.sort_values(by = 'R', ascending=False).reset_index(drop=True)
    else:
        repertoire = repertoire.sort_values(by = 'L', ascending=False).reset_index(drop=True)
    
    # for readability 
    repertoire = repertoire.rename(columns={'pitch_type': 'Pitch Type'})
    
    repertoire[['R', 'L', 'Shape+_R', 'Shape+_L']] = repertoire[['R', 'L', 'Shape+_R', 'Shape+_L']].astype(str)  # Convert the entire column to string first
    repertoire.loc[repertoire['R'] > '0', 'R'] += '%'
    repertoire.loc[repertoire['L'] > '0', 'L'] += '%'

    repertoire[['Shape+_R', 'Shape+_L', 'R', 'L']] = repertoire[['Shape+_R', 'Shape+_L', 'R', 'L']].replace({'0': '--'})    

    repertoire.loc[repertoire['R'] == '--', 'Shape+_R'] = '--'

    repertoire.rename(columns={'Shape+_R': 'Shape+ R', 
                               'Shape+_L': 'Shape+ L'}, inplace=True)
    
    if display_mode == 'Scouting':
        repertoire.drop(columns = ['R', 'Shape+ R', 'L', 'Shape+ L'], inplace=True)
    else:
        repertoire.drop(columns = ['Velo', 'HB', 'iVB', 'Sct. Grade'], inplace=True)

    if 'Movement-Based Changeup' in repertoire['Pitch Type'].values:
        if len(repertoire) < 5:
            repertoire.loc[repertoire['Pitch Type'] == 'Movement-Based Changeup', 'Pitch Type'] = 'Movement-Based\n Changeup'
        else:
            repertoire.loc[repertoire['Pitch Type'] == 'Movement-Based Changeup', 'Pitch Type'] = 'M-B Change'
    if 'Velo-Based Changeup' in repertoire['Pitch Type'].values:
        if len(repertoire) < 5:
            repertoire.loc[repertoire['Pitch Type'] == 'Velo-Based Changeup', 'Pitch Type'] = 'Velo-Based\n Changeup'
        else:
            repertoire.loc[repertoire['Pitch Type'] == 'Velo-Based Changeup', 'Pitch Type'] = 'V-B Change'

    return repertoire

def plot_repertoire(pitcher, year, classified_pitch_data):
         
    # get all pitches from this year
    df = classified_pitch_data.query('pitcher == @pitcher and game_year == @year').copy()
    
    # pitcher handedness
    hand = df.p_throws.values[0]
    
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
    df = df.sort_values('true_pitch_type', key=lambda x: (-x.map(x.value_counts())))
    
    # all pitch shapes
    HB = list(df.pfx_x)
    iVB = list(df.pfx_z)
    velo = list(df.release_speed)
    
    # make plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=-0.12, right=0.88, top=1, bottom=-0.05)
    ax.scatter(velo, HB, iVB, c=df.color)
    
    # find bbox for legend    
    def perceived_width(velo, hb, theta, phi):
        
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        
        if hand == 'R':
            x = (20 - hb)/40
        else:
            x = (hb - 20)/40
        
        y = (100 - velo)/30

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Calculate perceived width using the dot product
        return -x * sin_theta + y * cos_theta
            
    l_max = 0
    r_max = 0
    for i in range(len(velo)):
        w = perceived_width(velo[i], HB[i], 40, 20)
        if w > r_max:
            r_max = w
        if w < l_max:
            l_max = w
        
    if np.abs(l_max) > r_max:
        legend_x = {'R': 0.62, 'L': 0}
    else:
        legend_x = {'R': 0, 'L': 0.62}
    
    # make legend handles and labels
    repertoire = list(df.true_pitch_type.unique())
    try:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colordict[pitch], markersize=10) for pitch in repertoire]
    except KeyError:
        for i in range(len(repertoire)):
            if repertoire[i] in ('Movement-Based Changeup', 'Velo-Based Changeup'):
                repertoire[i] = repertoire[i].split()[0] + '\n ' + repertoire[i].split()[1]
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colordict[pitch], markersize=10) for pitch in repertoire]
    
    # sort handles and labels
    legend_labels = repertoire
    sorted_handles_labels = [(handles[i], legend_labels[i]) for i in range(len(legend_labels))]
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    
    # make legend
    legend_y =  0.75 - len(repertoire)*0.01
    
    if hand == 'R':
        legend = ax.legend(sorted_handles, sorted_labels, loc = 'center left', bbox_to_anchor=(legend_x[hand], legend_y), fontsize = 15)
    else:
        legend = ax.legend(sorted_handles, sorted_labels, loc = 'center left', bbox_to_anchor=(legend_x[hand], legend_y), fontsize = 15)

    ax.add_artist(legend)
    
    # set title
    # fig.suptitle('Movement Profile            ', fontsize=32, weight='bold')
    ax.set_xlabel('Velo', fontsize=25, labelpad=20)
    ax.set_ylabel('HB', fontsize=25, labelpad=20)
    ax.set_zlabel('iVB', fontsize=25, labelpad=20)
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
    # poly = Poly3DCollection(faces, alpha=0.8, facecolor = 'white', edgecolor='red')
    # ax.add_collection3d(poly)
    
    # set viewing angle
    ax.view_init(elev=20, azim=-40)
    
    return fig, ax, colordict
        
def get_ovr(names, year, verbose=True):
    
    df = pd.DataFrame(index = names, dtype=float)
    
    for name in names:
        pitcher_data, grades, xrv, performance_data = get_data(classified_pitch_data, pitcher_grades, name, year)
        if pitcher_data == 'fail':
            df.loc[name, '4S+'] = 0
            df.loc[name, 'Proj ERA'] = 0
            continue
        df.loc[name, '4S+'] = pitcher_data['4S+']
        df.loc[name, 'Proj ERA'] = performance_data.loc[2, 'SIERA']
    
    df.sort_values(by = '4S+', ascending=False, inplace=True)
    df['4S+'] = df['4S+'].astype(int)            
    df[['4S+', 'Proj ERA']] = df[['4S+', 'Proj ERA']].replace(0, '--')
    
    if verbose:
        print()
        print(df)
    else:
        return df
   
def top_ten(year, n = 10, starters=True, bottom=False):
    
    sp = 0
    if starters:
        sp = 1
        
    cond = (pitcher_grades['starter'] == sp) & (pitcher_grades['game_year'] == year)
    
    if bottom:
        top_names = pitcher_grades.loc[cond, 'Name'].tail(n)
    else:
        top_names = pitcher_grades.loc[cond, 'Name'].head(n)
    
    get_ovr(top_names, year)

def pitcher_profile(name, year, display_mode = 'Scouting'):

    ipython = get_ipython()
    ipython.run_line_magic('matplotlib', 'qt')    

    pitcher_data, grades, xrv, performance_data = get_data(classified_pitch_data, pitcher_grades, name, year)
    if pitcher_data == 'fail':
        return
    pitcher = pitcher_data['MLBAM_ID']
    
    # Create figure with subplots (7x6 layout)
    fig = plt.figure(figsize=(10, 10))
    
    gs = fig.add_gridspec(7, 6)
    ax_title = fig.add_subplot(gs[0, :])
    
    ax_polar = fig.add_subplot(gs[1:4, :3], polar=True)
    ax_rv = fig.add_subplot(gs[1, 3:])
    ax_perf = fig.add_subplot(gs[2, 3:])
    
    ax_viz = fig.add_subplot(gs[3:, 3:])
    ax_rep = fig.add_subplot(gs[4:, :3])
    ax_cred = fig.add_subplot(gs[6, :])
    plt.subplots_adjust(left=0.3, right=1.3, bottom=0, top=1, wspace=0.5, hspace=0)
    
    
    # Top stuff    
    name = pitcher_data['Name']
    year = pitcher_data['Year']
    hand = pitcher_data['Hand'] + 'H'
    if pitcher_data['SP'] == 1:
        pos_string = 'SP'
    else:
        pos_string = 'RP'
    ax_title.text(0.5,0.6,f"{name} ({hand} {pos_string}) - {year} Pitching Profile", fontsize=16, ha='center', va='center', weight='bold')
    ax_title.text(0.5,0.25,"By Johnny Nienstedt", fontsize=14, ha='center', va='center')
    ax_title.axis('off')
    
    
    
    #
    # 4S Grades
    #
    
    
    
    # Radar chart 
    
    categories = list(grades.keys())
    values = np.array(list(grades.values()))
    
    # Normalize values to a scale of 0 to 1
    normalized_values = values / 200 + 0.23
    
    # Create angles for each category (quartered layout)
    angles = [3*np.pi/4, 5*np.pi/4, 7*np.pi/4, np.pi/4]
    
    # Repeat first value to close the circular plot
    values = np.concatenate((normalized_values, [normalized_values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    # For each bar/section, create a custom colormap that goes from darker to lighter
    base_colors = {
        'green': '#006400',  # Dark green
        'red': '#8B0000',    # Dark red
        'orange': '#8B4500', # Dark orange
        'blue': '#00008B'    # Dark blue
    }
    
    light_colors = {
        'green': '#90EE90',  # Light green
        'red': '#FFB6C6',    # Light red
        'orange': '#FFE4B5', # Light orange
        'blue': '#87CEEB'    # Light blue
    }
    
    # Bar chart to mimic the quartered slices
    bars = ax_polar.bar(angles[:-1], values[:-1], width=np.pi / 2, 
                  color=['green', 'red', 'orange', 'blue'], alpha=0.5, edgecolor='black')
    
    # Create gradients for each section
    center_radius = 0.25
    for bar, color_name in zip(bars, ['green', 'red', 'orange', 'blue']):
        
        # Create custom colormap for this section
        cmap = LinearSegmentedColormap.from_list('custom', 
            [base_colors[color_name], light_colors[color_name]])
        
        # Get the number of gradient segments (more segments = smoother gradient)
        dr = 0.13
        r0 = center_radius
        r_max = bar.get_height()
        n_segments = int(np.ceil((r_max - r0)/dr))
        
        # Calculate the radii for each segment
        
        for i in range(n_segments):
            
            # initial radius
            radius = r0 + i * dr
            
            # Create a thin bar for this segment
            if i < n_segments - 1:
                ax_polar.bar(bar.get_x() + np.pi/4, dr, bottom=radius, 
                                       width=bar.get_width(), 
                                       color=cmap(i/n_segments), 
                                       edgecolor=None,
                                       alpha=0.5)
            else:
                dr = r_max - radius
                
                ax_polar.bar(bar.get_x() + np.pi/4, dr, bottom=radius, 
                                       width=bar.get_width(), 
                                       color=cmap(i/n_segments), 
                                       edgecolor=None,
                                       alpha=0.5)
    
    # Add black edges at the end
    bars = ax_polar.bar(angles[:-1], values[:-1], width=np.pi / 2, 
                       fill=False, edgecolor='black')
    
    
    
    # Add labels
    ax_polar.text(angles[0], values[0] + 0.05, categories[0] + '+', ha='right', va='bottom', fontsize=16, fontweight='bold')
    ax_polar.text(angles[1], values[1] + 0.05, categories[1] + '+', ha='right', va='top', fontsize=16, fontweight='bold')
    ax_polar.text(angles[2], values[2] + 0.05, categories[2] + '+', ha='left', va='top', fontsize=16, fontweight='bold')
    ax_polar.text(angles[3], values[3] + 0.05, categories[3] + '+', ha='left', va='bottom', fontsize=16, fontweight='bold')
    
    ax_polar.text(angles[0], values[0]/2 - 0.05, str(list(grades.values())[0]), ha='right', va='bottom', fontsize=19, fontweight='bold', color='black', path_effects=[path_effects.withStroke(linewidth=1, foreground='lightgray')])
    ax_polar.text(angles[1], values[1]/2 - 0.05, str(list(grades.values())[1]), ha='right', va='top', fontsize=19, fontweight='bold', color='black', path_effects=[path_effects.withStroke(linewidth=1, foreground='lightgray')])
    ax_polar.text(angles[2], values[2]/2 - 0.05, str(list(grades.values())[2]), ha='left', va='top', fontsize=20, fontweight='bold', color='black', path_effects=[path_effects.withStroke(linewidth=1, foreground='lightgray')])
    ax_polar.text(angles[3], values[3]/2 - 0.05, str(list(grades.values())[3]), ha='left', va='bottom', fontsize=20, fontweight='bold', color='black', path_effects=[path_effects.withStroke(linewidth=1, foreground='lightgray')])
    
    
    # Add central circle to mimic the style
    center_circle = plt.Circle((0, 0), center_radius, color='black', transform=ax_polar.transData._b, zorder=10)
    ax_polar.add_artist(center_circle)
    
    # Add overall rating in center
    ax_polar.text(13*np.pi/8,0.015, f'OVR\n{pitcher_data["4S+"]}', ha='center', va='center', fontsize=16, 
            fontweight='bold', color='white', zorder=11, bbox=dict(facecolor='black', edgecolor='none', boxstyle='circle,pad=0.4'), path_effects=[path_effects.withStroke(linewidth=1, foreground='white')])
    
    # Add title
    if np.max(list(grades.values())) > 140:
        ax_polar.text(np.pi/2, 1, "4S Grades", fontsize=14, fontweight='bold', ha='center', va='bottom')
    else:
        ax_polar.text(np.pi/2, 1, "4S Grades", fontsize=14, fontweight='bold', ha='center', va='center')

    # Remove the outer frame 
    ax_polar.set_ylim(0, 1)
    ax_polar.axis('off')
    
    ax_polar.set_xticks([])
    ax_polar.set_yticks([])
    
    
    #
    # RV Breakdown
    # 
    
    ax_rv.axis('off')
    if pitcher_data['SP'] == 1:
        n_pitches = 3000
    else:
        n_pitches = 1000
    
    rvs = list(n_pitches*np.array([xrv['Shape_RV'], xrv['Spot_RV'], xrv['Slot_RV'], xrv['Sequence_RV'], xrv['xRV']]))
    rvs = list((np.round(np.array(rvs)).astype(int)))
    for i in range(len(rvs)):
        if rvs[i] > 0:
            rvs[i] = '+' + str(rvs[i])
    
    ax_rv.text(0.5, 1, 'Full season run value from...', fontsize=14, ha='center', va='center', weight='bold')
    ax_rv.text(0.5, -0.05, f'*Full season for {pos_string} defined as {n_pitches} pitches', ha='center', va='center', fontsize=8)

    
    def get_text_width(word):
        font = ImageFont.load_default()
        bbox = font.getbbox(word)
        return bbox[2]
    
    words = ['Shape', 'Spot', 'Slot', 'Sequence', 'Total']
    widths = [get_text_width(word) for word in words]  # Get widths for all words
    
    dx = 30  # The spacing between words
    
    
    places = np.empty(5)
    places[0] = widths[0]/2
    places[1] = places[0] + widths[0]/2 + dx + widths[1]/2
    places[2] = places[1] + widths[1]/2 + dx + widths[2]/2
    places[3] = places[2] + widths[2]/2 + dx + widths[3]/2
    places[4] = places[3] + widths[3]/2 + dx + widths[4]/2
    
    table_width = np.sum(widths) + (len(words) - 1) * dx  # Calculate total table width including gaps
    
    # Cumulative width and placement adjustment
    available_width = 0.8  # Proportion of available width (can adjust if needed)
    
    # Normalize the cumulative widths to fit the available space (width = 1)
    places = places * available_width / table_width + 0.1
    
    for i, word in enumerate(words):
        ax_rv.text(places[i], 0.6, word, fontsize=12, ha='center', va='center')
        ax_rv.text(places[i], 0.3, rvs[i], fontsize=12, ha='center', va='center', weight='bold')
        
        if i < len(words) - 1:
            
            x = places[i] + (widths[i]/2 + dx/2) * available_width / table_width
            
            ax_rv.vlines(x, 0.1, 0.7, color='black', lw=1)
            
    ax_rv.set_xlim(0, 1)  # Keep fixed limits for the subplot
    ax_rv.set_ylim(0, 1)
    
    
    
    #
    # Performance data table
    #
    
    ax_perf.axis('tight')
    ax_perf.axis('off')
    table = ax_perf.table(cellText=performance_data.values, colLabels=performance_data.columns, colWidths = [0.2, 0.2, 0.2, 0.2], loc='center', cellLoc='center', bbox = [0.1, -0.5, 0.8, 1.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    ax_perf.set_xlim(0, 1) 
    ax_perf.set_ylim(0, 1)
    ax_perf.text(0.5, 0.97, 'Performance Metrics', ha='center', va='center', fontsize=14, weight='bold')
    
    
    
    
    
    #
    # Repertoire viz
    # 
    
    fig, ax, colordict = plot_repertoire(pitcher, year, classified_pitch_data)
    
    # Convert the figure to an image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    aspect_ratio = w / h
    scaling=2
    
    # Reshape with the scaled dimensions
    try:
        image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    except ValueError:
        image = np.frombuffer(buf, dtype=np.uint8).reshape(h*scaling, w*scaling, 4)
        
    image = image[:, :, :3]  # Convert RGBA to RGB
    
    # # Display the image in the target axes
    ax_viz.clear()    
    ax_viz.set_xlim((0,8))
    ax_viz.set_ylim((0,7))
    
    x1 = 0.3
    y1 = 0.9
    
    width = 8.8
    height = width/aspect_ratio
    
    x2 = x1 + width
    y2 = y1 + height
    
    ax_viz.imshow(image, extent=(x1,x2,y1,y2))
    ax_viz.text(4, 6.5, 'Movement Profile', fontsize=14, weight='bold', ha='center', va='center')
    ax_viz.axis('off')
    
    # Close the original figure
    plt.close(fig)
    
    
    
    
    
    
    #
    # Repertoire table
    # 
    
    ax_rep.axis('tight')
    ax_rep.axis('off')
    ax_rep.set_xlim((0,1))
    ax_rep.set_ylim((0,1))
    repertoire = grade_repertoire(pitcher, year, pitch_type_shape_grades, classified_pitch_data, display_mode=display_mode)
    
    if display_mode == 'Shape':
        table = ax_rep.table(cellText=repertoire.values, colLabels=repertoire.columns, colWidths = [0.3, 0.12, 0.23, 0.12, 0.23], loc='right', cellLoc='center', bbox = [0, 0.25, 1, 0.75])
    else:
        table = ax_rep.table(cellText=repertoire.values, colLabels=repertoire.columns, colWidths = [0.3, 0.18, 0.12, 0.14, 0.26], loc='right', cellLoc='center', bbox = [0, 0.25, 1, 0.75])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    ax_rep.set_title('Repertoire Grades', fontsize='14', weight='bold')
    
    pitch_col_idx = repertoire.columns.get_loc('Pitch Type')
    colordict['M-B Change'] = colordict['Movement-Based\n Changeup']
    colordict['V-B Change'] = colordict['Velo-Based\n Changeup']
    for idx, pitch in enumerate(repertoire['Pitch Type']):
            cell = table[idx + 1, pitch_col_idx]  # +1 because row 0 is header
            cell.set_facecolor([*mcolors.to_rgba(colordict[pitch])[:-1], 0.5])  # 0.5 is the alpha value, adjust between 0 and 1
    
    
    #
    # Credits
    #
    
    ax_cred.axis('off')
    ax_cred.text(0.5, 0.6, 'All data via MLB Advanced Media', ha='center', va='center', fontsize=10)
    ax_cred.text(0.5, 0.35, 'Documentation: github.com/johnnynienstedt/4S_Pitching', ha='center', va='center', fontsize=10)
    
    # Show the complete visualization
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()



'''
###############################################################################
############################## Pitcher Analysis ###############################
###############################################################################
'''
  
# get_ovr(['Logan Webb', 'Robbie Ray', 'Justin Verlander', 'Kyle Harrison', 
#          'Jordan Hicks', 'Hayden Birdsong', 'Landen Roupp'], 2024)

# top_ten(2024, starters=True)

# pitcher_profile('Jacob deGrom', 2024)

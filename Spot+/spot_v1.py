#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:51:35 2024

@author: johnnynienstedt
"""

#
# Spot+
#
# Johnny Nienstedt 12/27/24
#

# 
# This is the second leg of 4S pitching. Instead of using estimated target
# location to evaluate command, I will simply evaluate each pitch's location
# based on expected run value. I have no idea if this will be predictive, but
# we shall see.
#

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter





league_heatmaps = np.load('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Batting Analysis/SEAGER/league_heatmaps.npy')
classified_pitch_data = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Data/classified_pitch_data.csv')


def loc_rv(row, league_heatmaps):

    try:
        # Get situation
        b = int(row['balls'])
        s = int(row['strikes'])
        h = int(row['platoon'])
        
        # Input validation
        if not (0 <= b <= 3 and 0 <= s <= 2 and 0 <= h <= 1):
            return np.nan
        
        # Create location grids
        xlist = np.linspace(-15/13.5, 15/13.5, 30) + 1/27
        zlist = np.linspace(14/12, (14+32)/12, 35) + 1/27
        
        # Get pitch location
        px = float(row['plate_x'])
        pz = float(row['plate_z'])
        
        # Find nearest grid points
        x = xlist[np.abs(xlist - px).argmin()]
        z = zlist[np.abs(zlist - pz).argmin()]
        
        # Get indices for matrix
        ix = np.where(xlist == x)[0][0]
        iz = np.where(zlist == z)[0][0]
        
        # Return run value from league heatmaps
        return float(league_heatmaps[0, b, s, h, ix, iz])
    
    except (ValueError, KeyError, IndexError) as e:
        print(f"Error processing row: {e}")
        return np.nan

classified_pitch_data['loc_rv'] = classified_pitch_data.apply(lambda row: loc_rv(row, league_heatmaps), axis=1)


loc_grades = classified_pitch_data.groupby(['pitcher', 'player_name', 'game_year']).mean(numeric_only=True)['loc_rv'].reset_index()

counts = classified_pitch_data.groupby(['pitcher', 'player_name', 'game_year']).size().reset_index()

loc_grades['count'] = counts[0]
loc_grades['loc_rv'] = round(-2000*(loc_grades['loc_rv'] - loc_grades['loc_rv'].mean()), 2)




# bayesian projection with weakly informative normal prior
mu = 0
var = 1
n = loc_grades['count']
x = loc_grades['loc_rv']
s = 63**2
loc_grades['proj_loc'] = round((mu/var + n*x/s)/(1/var + n/s), 2)
loc_grades['proj_var'] = round(1/(1/var + n/s), 2)
loc_grades['proj_rv'] = round(loc_grades['proj_loc']/2000*loc_grades['count'], 2)




def yoy(df, colname, q, verbose=False):
    
    filtered_results = df[df['count'] > q].reset_index()
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
    

r = []
for i in np.arange(0, 2000, 100):
    r.append(yoy(loc_grades, 'proj_rv', i))
    
plt.figure()
plt.plot(np.arange(0, 2000, 100), r)
plt.ylim(bottom=0)
plt.show()




# plot prior against actual distribution
x = np.linspace(-20, 20, 1000)
prior = norm.pdf(x, mu, var)



# plt.figure()
# plt.hist(loc_grades['loc_rv'], 100)
# plt.plot(x, prior*700, label = 'Prior')
# plt.show()


loc_grades_2024 = loc_grades.query('game_year == 2024')[['pitcher', 'proj_loc', 'count']]

xRV = loc_grades_2024['proj_loc']
xRV_mean = xRV.mean()
xRV_std = xRV.std()
norm_grades = ((xRV - xRV_mean)/xRV_std + 10)*10

loc_grades_2024['Spot+'] = norm_grades.astype(int)


# plot posteriors
def plot_command(df, pitchers, include_prior = True):
    
    plt.figure()
    
    if include_prior:
        plt.plot(x, prior, label = 'Prior')
    
    if type(pitchers) == list:
        
        for pitcher in pitchers:
            
            name = pitcher.split(', ')[1] + ' ' +  pitcher.split(', ')[0]
            
            try:
                index = df.query('player_name == @pitcher').index[0]
                
            except IndexError:
                print('\n', pitcher, 'not found.')
                
            p = norm.pdf(x, df.loc[index, 'proj_rv'], df.loc[index, 'proj_var'])
            
            plt.plot(x, p, label = name)
            
    elif type(pitchers) == str:
        
        name = pitchers.split(', ')[1] + ' ' + pitchers.split(', ')[0]
        
        try:
            index = df.query('player_name == @pitchers').index[0]
            
        except IndexError:
            print('\n', pitchers, 'not found.')
            
        p = norm.pdf(x, df.loc[index, 'proj_rv'], df.loc[index, 'proj_var'])
        
        plt.plot(x, p, label = name)

    plt.xlim(-20, 20)
    plt.ylim(bottom=0)
    plt.title('2024 Posterior Command Distributions')
    plt.xlabel('True Talent Command (Location RV per 2000 pitches)')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper left')
    plt.show()



loc_grades_2024.to_csv('spot_grades.csv')





pitcher_grades_plus.query('count > 100').sort_values(by='2S+', ascending=False)
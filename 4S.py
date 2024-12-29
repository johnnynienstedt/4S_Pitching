#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:40:48 2024

@author: johnnynienstedt
"""

#
# 4S Pitching -- Combination of Shape, Spot, Slot, and Sequence
#
# Johnny Nienstedt 12/28/24
#

import pybaseball
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from IPython import get_ipython



# 4S grades
shape_grades = pd.read_csv('./shape_grades.csv')
spot_grades = pd.read_csv('./spot_grades.csv')

# combine
df = pd.merge(shape_grades, spot_grades, on='pitcher', how='inner')
df['2S+'] = (0.75*df['Shape+'] + 0.25*df['Spot+']).astype(int)

# combine with results from pybaseball
results = pybaseball.pitching_stats(2024, qual = 1)
results_df = pd.merge(results, df, on='Name', how='inner')

def pitcher_correlations(ind_colname, dep_colname, q=400):
        
    filtered_results = results_df[results_df.Pitches > q]
    
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
    
    
pitcher_correlations('Location+', 'BB%', q=1000)
pitcher_correlations('botCmd', 'BB%', q=1000)    
pitcher_correlations('Spot+', 'BB%', q=1000)

pitcher_correlations('Stuff+', 'K%', q=1000)    
pitcher_correlations('botStf', 'K%', q=1000)    
pitcher_correlations('Shape+', 'K%', q=1000)

pitcher_correlations('Pitching+', 'SIERA', q=1000)
pitcher_correlations('botOvr', 'SIERA', q=1000)
pitcher_correlations('2S+', 'SIERA', q=1000)




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

import numpy as np
import pybaseball
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from unidecode import unidecode
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression



'''
###############################################################################
################################# Import Data #################################
###############################################################################
'''

# 4S grades
shape_grades = pd.read_csv('Shape+/shape_grades.csv')
spot_grades = pd.read_csv('Spot+/spot_grades.csv')
slot_grades = pd.read_csv('Slot+/slot_grades.csv')
sequence_grades = pd.read_csv('Sequence+/sequence_grades.csv', index_col=0)

# combine
df = pd.merge(shape_grades, spot_grades, on=['pitcher', 'player_name', 'game_year'], how='inner')
df = pd.merge(df, slot_grades, on=['pitcher', 'player_name','game_year'], how='inner')
df = pd.merge(df, sequence_grades, on=['pitcher', 'player_name','game_year'], how='inner')

# combine with results from pybaseball
df['Name'] = df['Name'].apply(unidecode)
df['Season'] = df['game_year']
results = pybaseball.pitching_stats(2023, 2024, qual=1, ind=1)
results_df = pd.merge(results, df, on=['Name', 'Season'], how='outer')
sp_or_rp = pd.read_csv('Data/sp_or_rp.csv')[['Season', 'Name', 'starter']]
results_df = results_df.merge(sp_or_rp, on=['Name', 'Season'], how='outer')
results_df = results_df.dropna(subset='swstr')



'''
###############################################################################
############################# Predict Performance #############################
###############################################################################
'''

def normalize(data, desired_mean, desired_std, dtype='Int64'):
    
    m = data.mean()
    s = data.std()
    
    if dtype == 'Int64':
        normalized_data = (((data - m) / s) * desired_std + desired_mean).round().fillna(data).astype(dtype)
    else:
        normalized_data = (((data - m) / s) * desired_std + desired_mean).astype(dtype)
    
    return normalized_data

# linear regression
def multiple_linear_regression(data, xcols, ycol, q=None, significance_level=0.05, test_size=0.2, random_state=42):

    if q:
        df = data[data['Pitches'] > q].dropna(subset = xcols + [ycol]).copy()
    else:
        df = data.dropna(subset = xcols + [ycol]).copy()
    
    # Prepare data
    X = df[xcols]
    y = df[ycol]
                  
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Feature significance test
    f_scores, p_values = f_regression(X_scaled, y)
    
    # Select significant features
    significant_features = [
        col for col, p_val in zip(xcols, p_values) 
        if p_val <= significance_level
    ]
    
    if not significant_features:
        raise ValueError("No statistically significant features found")
    
    # Prepare final feature set
    X_selected = X[significant_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state
    )
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Collect results
    results = {
        'model': model,
        'significant_features': significant_features,
        'coefficients': dict(zip(significant_features, model.coef_)),
        'intercept': model.intercept_,
        'mean_squared_error': mse,
        'r2_score': r2,
        'feature_p_values': dict(zip(xcols, p_values))
    }
    
    return results


# describe preformance...
def describe(df, ycol, q=1000):
    print('\nDescribing ' + ycol + '...\n')
    
    shape_cols = ['swstr', 'foul', 'weak', 'topped', 'under', 'flr_brn', 'solid', 'barrel']
    spot_cols = ['loc_rv', 'proj_loc', 'pbp_variance']
    slot_cols = ['diff_rv', 'slot_rarity']
    sequence_cols = ['release_variance', 'repertoire_variance', 'sequence_rv']
    xcols = shape_cols + spot_cols + slot_cols + sequence_cols

    df[xcols] = normalize(df[xcols], 0, 1, dtype=float)
    l = 100
    x = np.linspace(1, 10**(-15), l)
    i = np.arange(0,l)
    r = [np.nan] * l
    for i, p in tqdm(enumerate(x)):
        try:
            results = multiple_linear_regression(df, xcols, ycol, significance_level=p, q=q)
            r[i] = results['r2_score']
        except ValueError:
            break
    
    best_r = np.nanmax(np.abs(r))
    best_i = list(np.abs(r)).index(best_r)
    best_p = x[best_i]
    
    results = multiple_linear_regression(df, xcols, ycol, significance_level=best_p, q=q)
    
    sig_cols = list(results['coefficients'].keys())
    coeff = list(results['coefficients'].values())
    rounded_coeff = np.round([100*x / sum(abs(x) for x in coeff) for x in coeff], 4)
    
    print('R-squared:', round(best_r, 3))

    print('\nWeights:')
    shape_coeff, spot_coeff, slot_coeff, sequence_coeff = (0, 0, 0, 0)
    shape_plus, spot_plus, slot_plus, sequence_plus = (0, 0, 0, 0)

    for i, col in enumerate(sig_cols):
        if col in shape_cols:
            shape_coeff += np.abs(rounded_coeff[i])
            shape_plus += coeff[i]
        if col in spot_cols:
            spot_coeff += np.abs(rounded_coeff[i])
            spot_plus += coeff[i]
        if col in slot_cols:
            slot_coeff += np.abs(rounded_coeff[i])
            slot_plus += coeff[i]
        if col in sequence_cols:
            sequence_coeff += np.abs(rounded_coeff[i])
            sequence_plus += coeff[i]
    
    print('Shape+: ' + str(round(shape_coeff, 1)))
    print('Spot+: ' + str(round(spot_coeff, 1)))
    print('Slot+: ' + str(round(slot_coeff, 1)))
    print('Sequence+: ' + str(round(sequence_coeff, 1)))

    pred_vals = sum([df[col] * coeff[i] for i, col in enumerate(sig_cols)])
    
    pop_mean = df[ycol].mean()
    pop_std = df.loc[df['Pitches'] > q, ycol].std()
    norm_vals = normalize(pred_vals, desired_mean=pop_mean, desired_std=pop_std, dtype=float)
    
    if ycol == 'SIERA':
        # df['Shape+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in shape_cols), 100, 10)
        df['Spot+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in spot_cols), 100, 10)
        # df['Slot+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in slot_cols), 100, 10)
        # df['Sequence+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in sequence_cols), 100, 10)
    
    return norm_vals
    
# and predict (spot goes in timeout)
def predict(df, ycol, q=1000):
    
    print('\nPredicting ' + ycol + '...\n')
    
    shape_cols = ['swstr', 'foul', 'weak', 'topped', 'under', 'flr_brn', 'solid', 'barrel']
    spot_cols = []
    slot_cols = ['diff_rv', 'slot_rarity']
    sequence_cols = ['sequence_rv']
    if ycol == 'BB%':
        spot_cols += ['proj_loc', 'pbp_variance']
        sequence_cols += ['release_variance', 'repertoire_variance']
        
    xcols = shape_cols + spot_cols + slot_cols + sequence_cols
    
    df[xcols] = normalize(df[xcols], 0, 1, dtype=float)
    l = 100
    x = np.linspace(1, 10**(-15), l)
    i = np.arange(0,l)
    r = [np.nan] * l
    for i, p in tqdm(enumerate(x)):
        try:
            results = multiple_linear_regression(df, xcols, ycol, significance_level=p, q=q)
            r[i] = results['r2_score']
        except ValueError:
            break
    
    best_r = np.nanmax(np.abs(r))
    best_i = list(np.abs(r)).index(best_r)
    best_p = x[best_i]
    
    results = multiple_linear_regression(df, xcols, ycol, significance_level=best_p, q=q)
    
    sig_cols = list(results['coefficients'].keys())
    coeff = list(results['coefficients'].values())
    rounded_coeff = np.round([100*x / sum(abs(x) for x in coeff) for x in coeff], 2)
    
    print('R-squared:', round(best_r, 3))

    print('\nWeights:')
    shape_coeff = 0
    spot_coeff = 0
    slot_coeff = 0
    sequence_coeff = 0
    for i, col in enumerate(sig_cols):
        if col in shape_cols:
            shape_coeff += np.abs(rounded_coeff[i])
        if col in spot_cols:
            spot_coeff += np.abs(rounded_coeff[i])
        if col in slot_cols:
            slot_coeff += np.abs(rounded_coeff[i])
        if col in sequence_cols:
            sequence_coeff += np.abs(rounded_coeff[i])
    
    print('Shape+: ' + str(round(shape_coeff, 1)))
    print('Spot+: ' + str(round(spot_coeff, 1)))
    print('Slot+: ' + str(round(slot_coeff, 1)))
    print('Sequence+: ' + str(round(sequence_coeff, 1)))
    
    pred_vals = sum([df[col] * coeff[i] for i, col in enumerate(sig_cols)])
    
    pop_mean = df[ycol].mean()
    pop_std = df.loc[df['Pitches'] > q, ycol].std()
    norm_vals = normalize(pred_vals, desired_mean=pop_mean, desired_std=pop_std, dtype=float)
    
    if ycol == 'SIERA':
        df['Shape+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in shape_cols), 100, 10)
        # df['Spot+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in spot_cols), 100, 10)
        df['Slot+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in slot_cols), 100, 10)
        df['Sequence+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in sequence_cols), 100, 10)

    
    return norm_vals

for ycol in ['SIERA', 'K%', 'BB%']:
    results_df['deserved_' + ycol] = describe(results_df, ycol)

for ycol in ['SIERA', 'K%', 'BB%']:
    results_df['predicted_' + ycol] = predict(results_df, ycol)

# finalize grades
results_df['4S+'] = normalize(-results_df['predicted_SIERA'], desired_mean=100, desired_std=10, dtype='Int64')

cols_of_interest = ['game_year', 'Name', '4S+', 'pitcher', 'starter', 'Pitches',
                    'Shape+', 'Spot+', 'Slot+', 'Sequence+', 'K%', 'BB%', 'SIERA', 
                    'deserved_K%', 'deserved_BB%', 'deserved_SIERA',
                    'predicted_K%', 'predicted_BB%', 'predicted_SIERA']
grades = results_df.loc[results_df['Pitches'] > 10, cols_of_interest]
grades = grades.drop_duplicates(subset = ['pitcher', 'game_year']).dropna(subset = '4S+')
grades = grades.sort_values(by = 'predicted_SIERA', ascending = True).reset_index(drop=True)
grades

grades.loc[grades['deserved_BB%']<0.02, 'deserved_BB%'] = 0.02
grades.loc[grades['deserved_BB%']>0.16, 'deserved_BB%'] = 0.16
grades.loc[grades['predicted_BB%']<0.02, 'predicted_BB%'] = 0.02
grades.loc[grades['predicted_BB%']>0.16, 'predicted_BB%'] = 0.16

grades['SIERA_diff'] = grades['SIERA'] - grades['predicted_SIERA']

grades[['SIERA', 'deserved_SIERA', 'predicted_SIERA']] = round(grades[['SIERA', 'deserved_SIERA', 'predicted_SIERA']], 2)
grades[['K%', 'BB%', 'deserved_K%','deserved_BB%', 'predicted_K%', 'predicted_BB%']] = round(100*grades[['K%', 'BB%', 'deserved_K%','deserved_BB%', 'predicted_K%', 'predicted_BB%']], 1)
# grades.to_csv('Data/pitcher_grades.csv', index=False)



'''
###############################################################################
################################ Analyze Model ################################
###############################################################################
'''

def pitcher_correlations(ind_colname, dep_colname, q=400, show=True):
        
    filtered_results = results_df[results_df.Pitches > q].dropna(subset = [ind_colname, dep_colname])
    
    y = filtered_results[dep_colname]
    x = filtered_results[ind_colname]

    m, b, r, p, e = stats.linregress(x, y)
    
    if show:
        fig, ax = plt.subplots()
    
        ax.scatter(x, y, s=3)
        ax.plot(x, m*x+b, '--k')
        ax.set_title('Same-season ' + ind_colname + ' vs. ' + dep_colname + ' (min. ' + str(q) + ' pitches)')
        ax.set_xlabel(ind_colname + ' (Pitcher Level)')
        ax.set_ylabel(dep_colname)
        
        # ax.set_aspect('equal', adjustable='box')
        
        left = x.min()
        right = x.max() - (x.max() - x.min())*0.2
        top = y.max()*0.9
        bot = y.min()*1.1
        
        years = filtered_results['Season'].astype(int).unique()
        year_string = ', '.join(map(str, years))
        
        if m > 0: 
            ax.text(left, top, '$R^2$ = ' + str(round(r**2, 2)))
            ax.text(right, bot, 'Seasons: ' + year_string, ha='center')
        else: 
            ax.text(right, top, '$R^2$ = ' + str(round(r**2, 2)))
            ax.text(left, bot, 'Seasons: ' + year_string, ha='left')
        
        plt.show()
    
    return r**2
    
# pitcher_correlations('Location+', 'BB%', q=1000)
# pitcher_correlations('botCmd', 'BB%', q=1000)
# pitcher_correlations('deserved_BB%', 'BB%', q=1000)

# pitcher_correlations('Stuff+', 'K%', q=1000)
# pitcher_correlations('botStf', 'K%', q=1000)
# pitcher_correlations('deserved_K%', 'K%', q=1000)

# pitcher_correlations('Stuff+', 'SIERA', q=1000)
# pitcher_correlations('botStf', 'SIERA', q=1000)
# pitcher_correlations('deserved_SIERA', 'SIERA', q=1000)

# pitcher_correlations('Stuff+', 'ERA', q=1000)
# pitcher_correlations('botStf', 'ERA', q=1000)
# pitcher_correlations('deserved_SIERA', 'ERA', q=1000)

# pitcher_correlations('Shape+', 'SIERA', q=1000)
# pitcher_correlations('Spot+', 'SIERA', q=1000)
# pitcher_correlations('Slot+', 'SIERA', q=1000)
# pitcher_correlations('4S+', 'SIERA', q=1000)

def residual_correlations(data, x1, x2, y, q=1000):
    """
    Plot and analyze correlations between a second predictor and residuals from first prediction.
    
    Parameters:
    data (DataFrame): Input dataframe containing all variables
    x1 (str): Name of primary independent variable
    x2 (str): Name of secondary independent variable to correlate with residuals
    y (str): Name of dependent variable
    q (int): Minimum number of pitches filter (default 400)
    
    Returns:
    float: R-squared value of the residual correlation
    """
    # Filter data
    filtered_data = data[data.Pitches > q].copy().dropna(subset=[x1, x2, y])
    
    # Calculate initial regression and residuals
    Y = filtered_data[y]
    X1 = filtered_data[x1]
    m1, b1, r1, p1, std_err1 = stats.linregress(X1, Y)
    
    
    # Calculate residuals
    predicted_y = m1 * X1 + b1
    residuals = Y - predicted_y
    
    # Calculate residual regression
    X2 = filtered_data[x2]
    m2, b2, r2, p2, std_err2 = stats.linregress(X2, residuals)
    
    # Create plot
    fig, ax = plt.subplots()
    ax.scatter(X2, residuals, s=3)
    ax.plot(X2, m2*X2+b2, '--k')
    ax.set_title(f'2024 {x2} vs. Residuals from {y}~{x1} (min. {q} pitches)')
    ax.set_xlabel(f'{x2} (Pitcher Level)')
    ax.set_ylabel(f'Residuals ({y} - predicted {y})')
    
    # Add R-squared text
    left = X2.min()
    right = X2.max() - (X2.max() - X2.min())*0.2
    top = residuals.max()*0.9
    
    if m2 > 0:
        ax.text(left, top, f'$R^2$ = {round(r2**2, 2)}')
    else:
        ax.text(right, top, f'$R^2$ = {round(r2**2, 2)}')
    
    plt.show()
    
    return p2, r2**2

# residual_correlations(results_df, 'Shape+', 'Spot+', 'SIERA')

def yoy(df, colname, q, verbose=False):
    
    filtered_results = df[df['Pitches'] > q].sort_values(by='pitcher').reset_index()
    filtered_results.dropna(subset = [colname], inplace=True)
    
    rows = []
    for i in range(len(filtered_results) - 1):
        if filtered_results['pitcher'][i + 1] == filtered_results['pitcher'][i]:
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
    
    if verbose:
        fig, ax = plt.subplots()
        plt.scatter(x, y, s=3)
        x1 = np.linspace(x.min(), x.max(), len(x))
        plt.plot(x1, m*x1+b, '--k')
        ax.set_title("YOY " + colname + " (Pitcher Level, min. " + str(q) + " Pitches Per Season)")
        ax.set_xlabel('2023 ' + colname)
        ax.set_ylabel('2024 ' + colname)
            
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
        
        ax.text((left + right)/2, bot, 'Trained on data from 2021-2022', ha='center', va='top', fontsize=8)
        
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
    
# yoy(results_df, colname='predicted_BB%', q=1000, verbose=True)

def project(df, xcol, ycol, q, verbose=False):
    
    
    if type(q) != list:
        filtered_results = df[df['Pitches'] > q].sort_values(by=['pitcher', 'game_year']).drop_duplicates(subset = ['pitcher', 'game_year']).reset_index()
    else:
        filtered_results = df.sort_values(by=['pitcher', 'game_year']).drop_duplicates(subset = ['pitcher', 'game_year']).reset_index().copy()
        filtered_results = filtered_results[
            ~((filtered_results['game_year']==2024) & (filtered_results['Pitches']<1000))]
        filtered_results = filtered_results[
            ~((filtered_results['game_year']==2023) & (filtered_results['Pitches'] < q[0]))]
        filtered_results = filtered_results[
            ~((filtered_results['game_year']==2023) & (filtered_results['Pitches'] > q[1]))]

    filtered_results = filtered_results.dropna(subset = [xcol, ycol]).reset_index()
    
    rows = []
    for i in range(len(filtered_results) - 1):
        if filtered_results['pitcher'][i + 1] == filtered_results['pitcher'][i]:
            c1 = filtered_results[xcol][i]
            c2 = filtered_results[ycol][i + 1]
            c3 = filtered_results['game_year'][i]
            rows.append({'Year1': c1,
                          'Year2': c2,
                          'game_year': c3})
            
            
    pairs = pd.DataFrame(rows)
    
    
    # make plot
    x = pairs['Year1']
    y = pairs['Year2']
    
    m, b, r, p, std_err = stats.linregress(x, y)
    
    if verbose:
        fig, ax = plt.subplots()
        plt.scatter(x, y, s=3)
        x1 = np.linspace(x.min(), x.max(), len(x))
        plt.plot(x1, m*x1+b, '--k')
        ax.set_title("Predicting " + ycol + " with " + xcol + "\n(Pitcher Level, min. " + str(q) + " Pitches Per Season)")
        ax.set_xlabel('2023 ' + xcol)
        ax.set_ylabel('2024 ' + ycol)
            
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
        
        ax.text((left + right)/2, bot, 'Trained on data from 2021-2022', ha='center', va='top', fontsize=8)
        
        # percent if necessary
        if '%' in xcol:
            if (x.max() - x.min()) > 0.1:
                plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{100*x:.0f}%'))
            else:
                plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{100*x:.1f}%'))
        
        if '%' in ycol:
            if (y.max() - y.min()) > 0.1:
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100*y:.0f}%'))
            else:
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100*y:.1f}%'))        
        
        plt.show()

    return r**2


# project(results_df, 'Location+', 'BB%', q=1000, verbose=True)
# project(results_df, 'botCmd', 'BB%', q=1000, verbose=True)
# project(results_df, 'predicted_BB%', 'BB%', q=1000, verbose=True)

# project(results_df, 'Stuff+', 'K%', q=1000, verbose=True)
# project(results_df, 'botStf', 'K%', q=1000, verbose=True)
# project(results_df, 'predicted_K%', 'K%', q=1000, verbose=True)

# project(results_df, 'Stuff+', 'SIERA', q=1000, verbose=True)
# project(results_df, 'botStf', 'SIERA', q=1000, verbose=True)
# project(results_df, 'predicted_SIERA', 'SIERA', q=1000, verbose=True)

# project(results_df, 'Stuff+', 'ERA', q=1000, verbose=True)
# project(results_df, 'botStf', 'ERA', q=1000, verbose=True)
# project(results_df, 'predicted_SIERA', 'ERA', q=1000, verbose=True)


def compare_models_rolling(data, ycol, delta=100):
    
    if ycol not in ['K%', 'BB%', 'SIERA', 'ERA']:
        print('\nOptions for ycol are K%, BB%, SIERA, or ERA.')
        return
    
    model_dict = {'Fangraphs': {'K%': 'Stuff+',
                                'BB%': 'Location+',
                                'SIERA': 'Pitching+',
                                'ERA': 'Pitching+'},
                  
                  'PitchingBot': {'K%': 'botStf',
                                  'BB%': 'botCmd',
                                  'SIERA': 'botOvr',
                                  'ERA': 'botOvr'}
                  }
    
    fangraphs_model = model_dict['Fangraphs'][ycol]
    pitchingbot_model = model_dict['PitchingBot'][ycol]
    if ycol == 'ERA':
        my_model = 'predicted_SIERA'
    else:
        my_model = 'predicted_' + ycol
    
    start = 2*delta
    stop = 3050
    volume_range = np.arange(start, stop+delta, delta)
    stuff_corr = [np.nan] * len(volume_range)
    bot_corr = [np.nan] * len(volume_range)
    my_corr = [np.nan] * len(volume_range)
    for index, pitches in enumerate(volume_range):
        stuff_corr[index] = project(data, fangraphs_model, ycol, q = [pitches-2*delta, pitches+2*delta])
        bot_corr[index] = project(data, pitchingbot_model, ycol, q = [pitches-2*delta, pitches+2*delta])
        my_corr[index] = project(data, my_model, ycol, q = [pitches-2*delta, pitches+2*delta])
    
    smoothed_stuff = gaussian_filter1d(stuff_corr, sigma=1)
    smoothed_bot = gaussian_filter1d(bot_corr, sigma=1)
    smoothed_4S = gaussian_filter1d(my_corr, sigma=1)
        
    fig, ax = plt.subplots()
    plt.plot(volume_range, np.sqrt(smoothed_stuff), color='green', label = '2023 ' + fangraphs_model)
    plt.plot(volume_range, np.sqrt(smoothed_bot), color='orange', label = '2023 ' + pitchingbot_model)
    plt.plot(volume_range, np.sqrt(smoothed_4S), color='red', label = '2023 4S_' + ycol)
    plt.title('Predicting ' + ycol + ' with Different Models', fontsize=14)
    plt.xlabel('Number of Pitches Thrown in 2023', fontsize=14)
    plt.ylabel('Pearson Correlation to 2024 ' + ycol, fontsize=12)
    plt.ylim((0,1))
    x_range = np.array(ax.get_xlim())
    plt.text(x_range.mean(), 0.05, '*minimum 1000 pitches thrown in 2024', ha='center', va='center')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return np.mean(my_corr)

# compare_models_rolling(results_df, 'K%')
# compare_models_rolling(results_df, 'BB%')
# compare_models_rolling(results_df, 'SIERA')
# compare_models_rolling(results_df, 'ERA')

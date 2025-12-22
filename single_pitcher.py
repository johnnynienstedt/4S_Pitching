# -*- coding: utf-8 -*-
"""
Created on Mon May  5 21:20:08 2025

@author: jnienstedt
"""

#
# Single pitcher 4S
#
# Johnny Nienstedt 5/5/2025
#



import math
import joblib
import pandas_gbq
import pydata_google_auth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from PIL import ImageFont
from numba import jit
from IPython import get_ipython
from unidecode import unidecode
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_data(pitcher, start_dt=None, end_dt=None, sf_level=None, game_type=None, manual_conditions=None):
    
    # credentials
    credentials = pydata_google_auth.get_user_credentials(
        [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/drive',
        ],
        auth_local_webserver=True,
    )
    
    # choose proper dataset
    if sf_level == 'intl':
        tbl = 'international'
    elif sf_level == 'am':
        tbl = 'amateur'
    else:
        tbl = 'pro'
        
    # select correct pitcher
    if type(pitcher) == str:
        if ',' in pitcher:
            pitcher_condition = f"ge.pitcher_name = '{pitcher}'"
        else:
            first = pitcher.split(' ')[0]
            last = pitcher.split(' ')[1]
            pitcher_condition = f"ge.pitcher_name = '{last}, {first}'"
    else:
        pitcher_condition = f"ge.pitcher_sfg_id = {pitcher}"
        
    # optional game date condition
    if start_dt is not None:
        if end_dt is not None:
            date_condition = f"ge.game_date BETWEEN '{start_dt}' AND '{end_dt}'"
        else:
            date_condition = f"ge.game_date > '{start_dt}'"
    else:
        if end_dt is not None:
            date_condition = f"ge.game_date < '{end_dt}'"
        else:
            date_condition = "ge.game_date is not NULL"

    # optional level condition
    if sf_level is not None:
        if type(sf_level) == str:
            if sf_level in ('intl', 'am'):
                level_condition = "1=1"
            else:
                level_condition = f"ge.sf_level = '{sf_level}'"
        elif type(sf_level) == tuple:
            level_condition = f"ge.sf_level in {sf_level}"
        else:
            raise ValueError("sf_level must be string or tuple of strings.")
    else:
        level_condition = "ge.sf_level is not NULL"
    
    # optional game_type condition
    if game_type is not None:
        if type(game_type) == str:
            if game_type == 'all':
                type_condition = '1 = 1'
            elif game_type == 'P':
                type_condition = 'ge.game_type in ("C", "D", "F", "L", "W")'
            else:
                type_condition = f"ge.game_type = '{game_type}'"
        elif type(game_type) == tuple:
            type_condition = f"ge.game_type in {game_type}"
        else:
            raise ValueError("game_type must be string or tuple of strings.")
    else:
        type_condition = "ge.game_type not in ('S', 'E', 'A')"
    if sf_level in ('intl', 'am'):
        type_condition = "1=1"
    
    # optional manual condition
    if manual_conditions is None:
        manual_conditions = '1=1'
    
    # handle duplicate names
    if type(pitcher) == str and sf_level != 'intl':
        
        id_query = f"""
        SELECT
        
        DISTINCT ge.pitcher_sfg_id 
        
        FROM `sfgiants-viewer.game_event.{tbl}` as ge
       
        WHERE 
        {pitcher_condition} and
        {date_condition} and
        {level_condition} and 
        {type_condition} and
        {manual_conditions}
        """
        
        name_ids = pandas_gbq.read_gbq(id_query,
                                       project_id="sfgiants-viewer",
                                       credentials=credentials,
                                       progress_bar_type=None).dropna()
                
        if len(name_ids) == 0:
            raise ValueError("No data on this pitcher for the given query.")
        elif len(name_ids) > 1:
            name_ids = list(name_ids['pitcher_sfg_id'].values)
            true_id = input(f"Multiple matches for {pitcher} in this query. Which of the following IDs is correct?\n{name_ids}\n\n")
        else:
            true_id = name_ids.values[0][0]
        pitcher_condition = f"ge.pitcher_sfg_id = {true_id}"
    
    conditions = {'start_dt': start_dt, 
                  'end_dt': end_dt, 
                  'sf_level': sf_level, 
                  'game_type': game_type}

    
    
    '''
    ###############################################################################
    ############################# Import & Clean Data #############################
    ###############################################################################
    '''

    four_s_query = f"""
    SELECT
    extract(day from ge.game_date) as day,
    extract(month from ge.game_date) as month,
    ge.year as game_year,
    ge.game_type,
    ge.game_pk,
    ge.home_team,
    ge.inning,
    ge.inning_plate_appearance as at_bat_number,
    ge.plate_appearance_pitch_number as pitch_number,
    ge.balls,
    ge.strikes,
    ge.pitcher_name as player_name,
    ge.pitcher_sfg_id as pitcher,
    ge.sf_level,
    ge.pitcher_team,
    CASE
      WHEN ge.hitter_side = 'left' THEN 'L'
      WHEN ge.hitter_side = 'right' THEN 'R'
      ELSE null
      END as stand,
    CASE
      WHEN ge.pitcher_throws = 'left' THEN 'L'
      WHEN ge.pitcher_throws = 'right' THEN 'R'
      ELSE null
      END as p_throws,
    CASE
      WHEN ge.pitch_type = 'four_seam' THEN 'FF'
      WHEN ge.pitch_type = 'sinker' THEN 'SI'
      WHEN ge.pitch_type = 'cutter' THEN 'FC'
      WHEN ge.pitch_type = 'slider' THEN 'SL'
      WHEN ge.pitch_type = 'sweeper' THEN 'ST'
      WHEN ge.pitch_type = 'curveball' THEN 'CU'
      WHEN ge.pitch_type = 'changeup' THEN 'CH'
      WHEN ge.pitch_type = 'splitter' THEN 'FS'
      WHEN ge.pitch_type = 'other' THEN 'OT'
      ELSE null
      END as pitch_type,
    ge.pitch_release_velocity as release_speed,
    ge.pitch_spin_rate as release_spin_rate,
    ge.pitch_x0 as release_pos_x,
    ge.pitch_y0 as release_pos_y,
    ge.pitch_z0 as release_pos_z,
    ge.pitch_extension as release_extension,
    ge.pitch_spin_axis as spin_axis,
    ge.pitch_plate_location_side as plate_x,
    ge.pitch_plate_location_height as plate_z,
    ge.pitch_horizontal_break as pfx_x,
    ge.pitch_induced_vertical_break as pfx_z,
    ge.pitch_vx0 as vx0,
    ge.pitch_vy0 as vy0,
    ge.pitch_vz0 as vz0,
    ge.pitch_ax0 as ax,
    ge.pitch_ay0 as ay,
    ge.pitch_az0 as az,
    ge.pitch_zone_time as ttp,
    ge.pitch_vertical_release_angle as release_angle_v,
    ge.pitch_horizontal_release_angle as release_angle_h,
    ge.pitch_vertical_approach_angle as VAA,
    ge.pitch_horizontal_approach_angle as HAA,
    
    ge.is_last_pitch_plate_appearance,
    ge.is_strikeout,
    ge.is_base_on_balls,
    ge.is_ball_in_play,
    ge.hit_exit_angle,

    90 - he.arm_slot_degree_at_release as arm_angle,
    he.rubber_position_in_inch as rubber_pos
    
    FROM `sfgiants-viewer.game_event.{tbl}` as ge
    LEFT JOIN `sfgiants-analyst.pitcher_biomechanics.hawkeye_artifacts_removed` as he
    ON he.pitch_uid = ge.trackman_pitch_uid
    
    WHERE 
    {pitcher_condition} and
    {date_condition} and
    {level_condition} and 
    {type_condition} and
    {manual_conditions}

    """
    
    all_pitch_data = pandas_gbq.read_gbq(four_s_query,
                                         project_id="sfgiants-viewer",
                                         credentials=credentials)
    
    if len(all_pitch_data) == 0:
        raise ValueError("No data for given query.")
        
    # fill missing values
    if sf_level in ('intl', 'am'):
        all_pitch_data['pitcher'] = all_pitch_data['pitcher'].fillna(999999)
        all_pitch_data['game_pk'] = all_pitch_data['game_pk'].fillna(1)
        all_pitch_data['game_type'] = all_pitch_data['game_type'].fillna('R')
        all_pitch_data['is_ball_in_play'] = all_pitch_data['is_ball_in_play'].fillna(False)
        
    if game_type == 'all':
        all_pitch_data['pitcher'] = all_pitch_data['pitcher'].fillna(999999)
        all_pitch_data['game_pk'] = all_pitch_data['game_pk'].fillna(1)
        all_pitch_data['game_type'] = all_pitch_data['game_type'].fillna('X')

    # get macro results
    pitcher = all_pitch_data['pitcher'].values[0]
    bf_per_g = all_pitch_data.groupby(['game_year', 'month', 'day'])['is_last_pitch_plate_appearance'].sum().mean()
    starter = 1 if bf_per_g > 12 else 0
    tbf = sum(all_pitch_data['is_last_pitch_plate_appearance'].dropna())
    k_pct = sum(all_pitch_data['is_strikeout'].dropna()) / tbf
    bb_pct = sum(all_pitch_data['is_base_on_balls'].dropna()) / tbf
    gbs = sum((all_pitch_data['is_ball_in_play']) & (all_pitch_data['hit_exit_angle'] < 10))
    fbs = sum((all_pitch_data['is_ball_in_play']) & (all_pitch_data['hit_exit_angle'].between(25, 50)))
    pus = sum((all_pitch_data['is_ball_in_play']) & (all_pitch_data['hit_exit_angle'] > 50))
    
    bip = (gbs - fbs - pus) / tbf
    pm = -1 if bip > 0 else 1
    siera = 6.645 - 16.986 * k_pct + 11.434 * bb_pct - 1.858 * bip + 7.653 * k_pct**2 + pm * 6.664 * bip**2 + 10.130 * k_pct * bip - 5.195 * bb_pct * bip

    results_dict = {'pitcher': pitcher,
                    'starter': starter,
                    'K%': k_pct,
                    'BB%': bb_pct,
                    'SIERA': siera}
        
    results_data = pd.DataFrame([results_dict])
        
    if len(results_data) == 0:
        print("\nWARNING: Unable to pull results for this query.")
    
    # ensure necessary data is present
    nececssary_cols = ['day', 'month', 'game_year', 'sf_level', 'game_pk', 
                       'pitcher_team', 'home_team', 'inning',
                       'at_bat_number','pitch_number', 'balls', 'strikes', 
                       'player_name', 'pitcher', 'stand', 'p_throws', 
                       'release_speed', 'release_pos_x', 
                       'release_pos_y', 'release_pos_z', 'release_extension',
                       'plate_x', 'plate_z', 'pfx_x', 'pfx_z', 'vx0', 
                       'vy0', 'vz0', 'ax', 'ay', 'az', 'release_angle_v', 
                       'release_angle_h', 'VAA', 'HAA']
    
    clean_data = all_pitch_data.dropna(subset = nececssary_cols).reset_index(drop=True)
    
    # ensure we have data
    if len(clean_data) > 1:
        pitch_data = clean_data.copy()
    else:
        raise ValueError("One or more necessary columns was missing.")
    
    # sort chronologically and remove duplicate rows
    sort_cols = ['game_year', 'month', 'day', 'game_pk', 'inning', 'at_bat_number', 'pitch_number']
    pitch_data = pitch_data.sort_values(by=sort_cols, ascending=True).drop_duplicates(subset=sort_cols, ignore_index=True)
    
    # flip axis for LHP so that +HB = arm side, -HB = glove side
    mirror_cols = ['release_pos_x', 'pfx_x', 'vx0', 'ax', 'release_angle_h', 'HAA']
    pitch_data.loc[pitch_data['p_throws'] == 'L', mirror_cols] = -pitch_data.loc[pitch_data['p_throws'] == 'L', mirror_cols]
    pitch_data['spin_axis'] = np.where(pitch_data['p_throws'] == 'L', 360 - pitch_data['spin_axis'], pitch_data['spin_axis'])
    
    # remove physically impossible arm angles
    pitch_data = pitch_data[pitch_data['arm_angle'].between(-65, 100) | pitch_data['arm_angle'].isna()]
    
    # ensure each pitcher has only one name
    id_to_name = pitch_data.groupby('pitcher')['player_name'].first()
    pitch_data['player_name'] = pitch_data['pitcher'].map(id_to_name)
    
    
    
    '''
    ###############################################################################
    ########################## Calculate More Parameters ##########################
    ###############################################################################
    '''
    
    # platoon rate
    pitch_data['platoon'] = np.where(pitch_data['p_throws'] == pitch_data['stand'], 0, 1)
    
    # arm angle_adj. spin axis (not currently used - may be used to help classify)
    pitch_data['adj_spin_axis'] = pitch_data['spin_axis'] - pitch_data['arm_angle'] - 90
    pitch_data['adj_spin_axis'] = np.where(pitch_data['adj_spin_axis'] < 0, 360 + pitch_data['adj_spin_axis'], pitch_data['adj_spin_axis'])
    
    # tilt
    pitch_data['tilt_obs'] = np.tan(pitch_data['az'] / pitch_data['ax'])
        
    # rotate accel
    ax = pitch_data['ax']
    ay = pitch_data['ay']
    az = pitch_data['az']
    x1 = pitch_data['release_pos_x']
    y1 = pitch_data['release_pos_y']
    z1 = pitch_data['release_pos_z']
    x2 = pitch_data['plate_x']
    y2 = 17 / 2 /12
    z2 = pitch_data['plate_z']
    
    theta = np.arctan((x2 - x1) / -(y2 - y1))
    phi = np.arctan((z2 - z1) / np.sqrt((x2-x1)**2 + (y2-y1)**2))

    pitch_data['av'] = (- np.sin(theta) * np.sin(phi) * ax
                        + np.cos(theta) * np.sin(phi) * ay
                        + np.cos(phi)                 * az)
    
    pitch_data['ah'] = (  np.cos(theta)               * ax
                        + np.sin(theta)               * ay)
    
    
    
    '''
    ###############################################################################
    ############################# Classify Pitch Types ############################
    ###############################################################################
    '''
    
    classified_pitch_data = pitch_data.copy()
    
    # function for determining repertoires
    def get_repertoire(data, pitcher, year='all'):
        df = data.copy()
    
        # number of pitches thrown
        n = len(df)
        if n == 0:
            raise AttributeError('No data for this pitcher & year(s).')
    
        pitch_type_df = df.groupby('pitch_type').agg(
            velo=('release_speed', 'mean'),
            hb=('pfx_x', 'mean'),
            ivb=('pfx_z', 'mean'),
            count=('release_speed', 'count'),
            platoon=('platoon', 'mean'),
            spin=('release_spin_rate', 'mean'),
            spin_axis=('adj_spin_axis', 'mean')
        ).sort_values(by='ivb', ascending=False)
    
        # get sinkers and 4-seamers for pitch shape baseline
        baseline = ['velo', 'hb', 'ivb']
        try:
            ff_baseline = pitch_type_df.loc['FF', baseline]
            if ff_baseline['hb'] < -2:
                ct_baseline = ff_baseline
                ff_baseline = ct_baseline + [3, 8, 5]
        except KeyError:
            try:
                si_baseline = pitch_type_df.loc['SI', baseline]
                if si_baseline['hb'] < 11 and si_baseline['ivb'] > 15:
                    ff_baseline = si_baseline
                else:
                    ff_baseline = si_baseline + [1, -5, 8]
            except KeyError:
                try:
                    ct_baseline = pitch_type_df.loc['FC', baseline]
                    if ct_baseline['ivb'] > 6:
                        ff_baseline = ct_baseline + [3, 8, 5]
                    else:
                        ff_baseline = ct_baseline + [6, 8, 10]
                except KeyError:
                    df['true_pitch_type'] = pd.NA
                    return df['true_pitch_type']
    
        # build si_baseline same logic as original
        si_baseline = ff_baseline + [-1, 6, -8]
    
        ffvel, ffh, ffv = ff_baseline
        sivel, sih, siv = si_baseline
    
        # pitch archetypes
        pitch_archetypes = np.array([
            [ffh, 20, ffvel],               # Riding Fastball
            [ffh, 11, ffvel],               # Fastball
            [sih, siv, sivel],              # Sinker
            # [-2, 13, ffvel-2],              # Cut-Fastball
            [-3, 8, ffvel - 3],             # Cutter
            [-1, 0, ffvel - 9],             # Gyro Slider
            [-8, -3, ffvel - 9],            # Two-Plane Slider
            [-7, 10, ffvel - 12],           # Carry Slider
            [-16, 1, ffvel - 14],           # Sweeper
            [-16, -6, ffvel - 15],          # Slurve
            [-8, -12, ffvel - 15],          # Curveball
            [-8, -12, ffvel - 22],          # Slow Curve
            [ffh + 2, siv - 5, sivel - 4],  # Movement-Based Changeup
            [ffh + 2, siv - 5, sivel - 5],  # Velo-Based Changeup
            # [sih, siv - 10, sivel - 8]      # Screwball
        ])
    
        # pitch names (note: kept same as original; there is an extra "Knuckleball" name)
        pitch_names = np.array([
            'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider', 'Carry Slider',
            'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup',
            'Velo-Based Changeup', 'Knuckleball'
        ])
    
        # --- Parameters you requested ---
        duplicates = 2                    # allow up to 2 copies per archetype (original + 1 duplicate)
        penalty_schedule = [0, 50]        # penalty for copy index 0,1 (we recommended this)
        alpha = 0.5                        # count exponent for scaling penalty
    
        # Build pitch shape matrix (n_pitch_types x 3)
        pitch_types = list(pitch_type_df.index)
        shapes = pitch_type_df[['hb', 'ivb', 'velo']].values  # shape = (P, 3)
        counts = pitch_type_df['count'].values                # shape = (P,)
    
        # Base distance matrix: (P, A) where A = number of archetypes
        base_archetypes = pitch_archetypes  # (A, 3)
        A = base_archetypes.shape[0]
    
        # compute euclidean distances from each pitch to each archetype (vectorized)
        # result `dists_base` shape = (P, A)
        # careful broadcasting: shapes[:, None, :] - base_archetypes[None, :, :]
        dists_base = np.linalg.norm(shapes[:, None, :] - base_archetypes[None, :, :], axis=2)
    
        # Expand archetypes (allow duplicates) and build corresponding penalty array
        # expanded_archetypes shape = (A * duplicates, 3) but we only need distances, so expand base dists
        # penalty_per_column length = A * duplicates
        penalty_per_column = np.tile(np.array(penalty_schedule), A)  # repeats [0,50] A times => length A*duplicates
        # Create expanded distance matrix: repeat each archetype column `duplicates` times
        dists_expanded = np.repeat(dists_base, repeats=duplicates, axis=1)  # shape = (P, A*duplicates)
    
        # Apply count-scaled penalty: for pitch i, add penalty_per_column * (count[i] ** alpha)
        # penalty_per_column broadcasts across rows
        count_scaler = (counts ** alpha).reshape(-1, 1)  # (P,1)
        cost_matrix = dists_expanded + (penalty_per_column.reshape(1, -1) * count_scaler)
    
        # Solve assignment: we have P rows (pitch types) and A*duplicates columns (expanded archetypes)
        # Hungarian returns optimal 1-to-1 matching for rows -> columns (rows <= cols assumed; that's true here)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
        # map column index back to archetype index (original archetype index)
        assigned_archetype_index = (col_ind // duplicates).astype(int)  # integer division
        
        # Build a dict mapping pitch_type -> assigned archetype index & store distance for later tie-breaks if needed
        assigned_map = {}
        # Also keep the base (non-penalized) distance for that assigned archetype (use dists_base)
        assigned_base_dist = {}
        for r, arche_idx in zip(row_ind, assigned_archetype_index):
            ptype = pitch_types[r]
            assigned_map[ptype] = int(arche_idx)
            assigned_base_dist[ptype] = float(dists_base[r, arche_idx])
    
        # Initialize series of results
        df['true_pitch_type'] = pd.NA
    
        # First apply OT special-case exactly as before
        for ptype, group in pitch_type_df.iterrows():
            if ptype == 'OT':
                if group['count'] >= 25:
                    pitch_name = 'Knuckleball'
                else:
                    pitch_name = pd.NA
                df.loc[df['pitch_type'] == ptype, 'true_pitch_type'] = pitch_name
    
        # Assign preliminary names according to assignment (skip OT since handled)
        # Keep track of which pitch_names have been assigned already (to prevent post-processing duplicates)
        assigned_name_to_ptype = {}
        for ptype in pitch_types:
            if ptype == 'OT':
                continue
            if ptype not in assigned_map:
                # if not assigned (unlikely), set NaN
                df.loc[df['pitch_type'] == ptype, 'true_pitch_type'] = pd.NA
                continue
    
            arche_idx = assigned_map[ptype]
            # If arche_idx is out of bounds (shouldn't be), fallback to NaN
            if arche_idx < 0 or arche_idx >= len(pitch_names):
                df.loc[df['pitch_type'] == ptype, 'true_pitch_type'] = pd.NA
                continue
    
            pname = pitch_names[arche_idx]
            df.loc[df['pitch_type'] == ptype, 'true_pitch_type'] = pname
            assigned_name_to_ptype[pname] = ptype
    
        # --- Post-processing corrections (preserve logic, but DO NOT create duplicates) ---
        # We'll attempt each correction but check if it would create duplication. If it does, we skip the change.
    
        # Helper to test & apply a name change (only if it does not create a duplicate)
        def try_apply_change(ptype, new_name, absolute=False):
            # if new_name is NaN-like, apply directly
            if pd.isna(new_name):
                df.loc[df['pitch_type'] == ptype, 'true_pitch_type'] = pd.NA
                # remove old mapping if existed
                for k, v in list(assigned_name_to_ptype.items()):
                    if v == ptype:
                        del assigned_name_to_ptype[k]
                return True
            

            # If new_name already assigned and it's assigned to a DIFFERENT pitch_type...
            existing = assigned_name_to_ptype.get(new_name, None)
            if existing is not None and existing != ptype:
                if not absolute:
                    return False

                for k, v in list(assigned_name_to_ptype.items()):
                    if v == ptype:
                        del assigned_name_to_ptype[k]
                    if v == existing:
                        old_name = k
                        del assigned_name_to_ptype[k]
                
                assigned_name_to_ptype[new_name] = ptype
                assigned_name_to_ptype[old_name] = existing
                df.loc[df['pitch_type'] == ptype, 'true_pitch_type'] = new_name
                df.loc[df['pitch_type'] == existing, 'true_pitch_type'] = old_name
                return True
    
            # Otherwise apply the change: remove previous name mapping if present, add the new mapping
            # remove any old name pointing to this ptype
            for k, v in list(assigned_name_to_ptype.items()):
                if v == ptype:
                    del assigned_name_to_ptype[k]
            assigned_name_to_ptype[new_name] = ptype
            df.loc[df['pitch_type'] == ptype, 'true_pitch_type'] = new_name
            return True
    
        # Apply post-processing sanity checks
        for ptype, group in pitch_type_df.iterrows():
            if ptype == 'OT':
                continue
            current_name = df.loc[df['pitch_type'] == ptype, 'true_pitch_type'].iloc[0]
            # safety if current_name is NaN
            if pd.isna(current_name):
                continue
    
            velo = group['velo']
    
            # get baseline velocities we computed earlier: ffvel, sivel
            # Note: ffvel and sivel set higher in code above
            # Sinker <-> Movement-Based Changeup rules
            if current_name == 'Sinker':
                if ffvel - velo > 3:
                    try_apply_change(ptype, 'Movement-Based Changeup', absolute=True)
            elif current_name in ['Movement-Based Changeup', 'Velo-Based Changeup']:
                if ffvel - velo < 3:
                    try_apply_change(ptype, 'Sinker', absolute=True)
    
            # Movement vs Velo changeup resolution
            if current_name in ['Movement-Based Changeup', 'Velo-Based Changeup']:
                # fetch sivel - velo difference
                if sivel - velo > 5:
                    # candidate -> Velo-Based Changeup
                    try_apply_change(ptype, 'Velo-Based Changeup', absolute=False)
                elif sivel - velo <= 5:
                    try_apply_change(ptype, 'Movement-Based Changeup', absolute=False)
    
        return df['true_pitch_type']
    
    # iterate over pitcher-years and re-classify (should take about one minute)
    pitcher_years = classified_pitch_data.groupby('pitcher').size().reset_index()[['pitcher']]
    for _, row in pitcher_years.iterrows():
        pitcher = row['pitcher']
        mask = (classified_pitch_data['pitcher'] == pitcher)
        classified_pitch_data.loc[mask, 'true_pitch_type'] = get_repertoire(classified_pitch_data[mask],
                                                                            pitcher=pitcher)
    
    classified_pitch_data.dropna(subset=['true_pitch_type'], inplace=True)


    
    '''
    ###############################################################################
    ########################### Adjust VAA/HAA for Loc ############################
    ###############################################################################
    '''
    
    # pitch_classes = {'fastball': ['Riding Fastball', 'Fastball'],
    #                  'sinker':   ['Sinker'],
    #                  'breaking': ['Cutter', 'Gyro Slider', 'Two-Plane Slider', 'Carry Slider',  'Sweeper', 
    #                               'Slurve', 'Curveball', 'Slow Curve'],
    #                  'offspeed': ['Movement-Based Changeup', 'Velo-Based Changeup',
    #                               'Knuckleball']
    #                  }
    
    # aa_models_dict = joblib.load('Data/aa_models_dict.joblib')
    
    # for pitch_class in pitch_classes:
    
    #     # VAA
    #     mask = classified_pitch_data['true_pitch_type'].isin(pitch_classes[pitch_class])
    #     x = classified_pitch_data.loc[mask, ['release_speed', 'tilt_obs', 'plate_z']]
    #     y = classified_pitch_data.loc[mask, 'VAA']
    
    #     if len(x) == 0:
    #         continue
    
    #     model = aa_models_dict['VAA'][pitch_class]
    #     y_pred = model.predict(x)
    #     classified_pitch_data.loc[mask, 'loc_adj_VAA'] = y - y_pred
        
    #     # HAA
    #     mask = classified_pitch_data['true_pitch_type'].isin(pitch_classes[pitch_class])
    #     x = classified_pitch_data.loc[mask, ['release_speed', 'tilt_obs', 'plate_x']]
    #     y = classified_pitch_data.loc[mask, 'HAA']
    
    #     model = aa_models_dict['HAA'][pitch_class]
    #     y_pred = model.predict(x)
    #     classified_pitch_data.loc[mask, 'loc_adj_HAA'] = y - y_pred
    
    classified_pitch_data['loc_adj_VAA'] = classified_pitch_data['VAA']
    classified_pitch_data['loc_adj_HAA'] = classified_pitch_data['HAA']    
    
    
    
    '''
    ###############################################################################
    ########################### Add Fastball Baselines ############################
    ###############################################################################
    '''

    fastballs = {'Riding Fastball', 'Fastball', 'Sinker'}

    # baselines for all pitch types
    pitch_agg = (
        classified_pitch_data.groupby(['pitcher', 'true_pitch_type'])
        .agg(
            pitch_count=('release_speed', 'count'),
            avg_velocity=('release_speed', 'mean'),
            avg_pfx_x=('pfx_x', 'mean'),
            avg_pfx_z=('pfx_z', 'mean')
        )
        .reset_index()
    )

    # select fastballs only
    target_subset = pitch_agg[pitch_agg['true_pitch_type'].isin(fastballs)]

    # weighted avg
    def weighted_average(group, col):
        return (group[col] * group['pitch_count']).sum() / group['pitch_count'].sum()

    weighted_avgs = (target_subset.groupby(['pitcher'])
                     .apply(lambda g: pd.Series({'fastball_velo': weighted_average(g, 'avg_velocity'),
                                                 'fastball_pfx_x': weighted_average(g, 'avg_pfx_x'),
                                                 'fastball_pfx_z': weighted_average(g, 'avg_pfx_z')}),include_groups=False)
                     .rename(columns={
                         'fastball_velo': 'fb_velo',
                         'fastball_pfx_x': 'fb_hb',
                         'fastball_pfx_z': 'fb_ivb'})
                     .reset_index())

    # if no fastballs, use fastest pitch
    fallbacks = (
        pitch_agg.sort_values(['pitcher', 'avg_velocity'], ascending=[True, False])
        .drop_duplicates(['pitcher'])
        .rename(columns={
            'avg_velocity': 'fb_velo',
            'avg_pfx_x': 'fb_hb',
            'avg_pfx_z': 'fb_ivb'})
        [['pitcher', 'fb_velo', 'fb_hb', 'fb_ivb']])

    # clean and merge
    final = pd.merge(fallbacks, weighted_avgs, on=['pitcher'], how='left', suffixes=('_fallback', '_weighted'))

    for col in ['fb_velo', 'fb_hb', 'fb_ivb']:
        if col not in final.columns:
            final[col] = final[f"{col}_weighted"].combine_first(final[f"{col}_fallback"])
    final_fastball_metrics = final[['pitcher', 'fb_velo', 'fb_hb', 'fb_ivb']].copy()
    final_fastball_metrics[['fb_velo', 'fb_hb', 'fb_ivb']] = final_fastball_metrics[['fb_velo', 'fb_hb', 'fb_ivb']].round(1)
    classified_pitch_data = classified_pitch_data.merge(final_fastball_metrics, how='left', on=['pitcher'])

    # calculate differentials
    classified_pitch_data['velo_diff'] = classified_pitch_data['release_speed'] - classified_pitch_data['fb_velo']
    classified_pitch_data['hb_diff'] = classified_pitch_data['pfx_x'] - classified_pitch_data['fb_hb']
    classified_pitch_data['ivb_diff'] = classified_pitch_data['pfx_z'] - classified_pitch_data['fb_ivb']
    
    return classified_pitch_data, results_data, conditions
classified_query_data, results_data, conditions = get_data(pitcher='Ron Marinaccio',
                                                           start_dt='2025-01-01',
                                                           end_dt='2025-12-31',
                                                           sf_level=None,
                                                           game_type=None)

def grade_shape(classified_pitch_data):
    
    # assign run values for each outcome
    rv = joblib.load('Data/outcome_rvs')

    # columns of interest
    x_cols = ['ttp', 'ah', 'av', 'release_pos_x', 'release_pos_z']
    outcomes = list(rv.keys())
    pred_cols = ['predicted_' + x for x in outcomes]
    display_cols = x_cols + pred_cols + ['release_speed', 'pfx_x', 'pfx_z']

    # train in groups
    pitch_classes = {'fastball': ['Riding Fastball', 'Fastball'],
                     'sinker':   ['Sinker'],
                     'breaking': ['Cutter', 'Gyro Slider', 'Two-Plane Slider', 'Carry Slider',  'Sweeper', 
                                  'Slurve', 'Curveball', 'Slow Curve'],
                     'offspeed': ['Movement-Based Changeup', 'Velo-Based Changeup',
                                  'Knuckleball']
                     }

    # store models for analysis
    models_dict = joblib.load('Data/shape_models.joblib')

    grading_data = classified_pitch_data.copy()

    for pitch_class in pitch_classes:
        
        if pitch_class == 'breaking':
            x_cols += ['velo_diff', 'hb_diff', 'ivb_diff']
        
        pitch_types = pitch_classes[pitch_class]
        p_grading_data = grading_data[grading_data.true_pitch_type.isin(pitch_types)]
                
        for outcome in outcomes:
            for hand, handedness in enumerate(['same-handed', 'opposite-handed']):
                eval_X = p_grading_data.loc[p_grading_data.platoon == hand, x_cols]
                calibrated_model = models_dict[handedness][pitch_class][outcome]
                grading_data.loc[eval_X.index, 'predicted_' + outcome] = calibrated_model.predict_proba(eval_X)[:, 1]
                   
    # no longer calculate raw Shape+ here
    # # use Shape+ coefficients or RVs?
    # # coeff = joblib.load('Data/SIERA_prediction_weights')['coefficients']
    # # grading_data['shape_rv'] = -sum((coeff[x] * grading_data['predicted_' + x] for x in outcomes))
    # grading_data['shape_rv'] = -sum((rv[x] * grading_data['predicted_' + x] for x in outcomes))
    # shape_rv_mean = joblib.load('Data/shape_rv_mean')
    # grading_data['shape_rv'] = grading_data['shape_rv'] - shape_rv_mean


    '''
    ###############################################################################
    ########################### By Type and Handedness ############################
    ###############################################################################
    '''

    grouped = grading_data.groupby(['pitcher', 'true_pitch_type', 'platoon'])[display_cols]
    repertoire = grouped.mean().reset_index().copy()
    
    counts = grouped.size().rename("count")
    repertoire = repertoire.merge(counts, on=['pitcher', 'true_pitch_type', 'platoon'])
    
    platoon_rate = grading_data['platoon'].mean()
    
    percentages = grouped.size() / len(grading_data)
    
    repertoire['percent'] = (
        repertoire.set_index(['pitcher', 'true_pitch_type', 'platoon']).index.map(percentages)
    )
    
    repertoire['percent'] = (
        repertoire['percent'] * 100 /
        repertoire['platoon'].map({0: 1 - platoon_rate, 1: platoon_rate})
    )
    
    # for convenience, ensure count column stays early in column order
    cols = list(repertoire.columns)
    cols.insert(4, cols.pop(cols.index('count')))
    repertoire = repertoire[cols]

    # rename
    repertoire.rename(columns={col: col.replace('predicted_', '') for col in repertoire.columns}, inplace=True)



    '''
    ###############################################################################
    ################################ By Pitch Type ################################
    ###############################################################################
    '''

    grouped = grading_data.groupby(['pitcher', 'true_pitch_type'])[display_cols]
    pt_shape_grades = grouped.mean().reset_index().copy()
    
    counts = grouped.size().rename("count")
    pt_shape_grades = pt_shape_grades.merge(counts, on=['pitcher', 'true_pitch_type'])
    
    # for convenience, ensure count column stays early in column order
    cols = list(pt_shape_grades.columns)
    cols.insert(2, cols.pop(cols.index('count')))
    pt_shape_grades = pt_shape_grades[cols]



    '''
    ###############################################################################
    ############################### By Pitcher Only ###############################
    ###############################################################################
    '''
    
    grouped = grading_data.groupby(['pitcher'])[display_cols]
    shape_grades = grouped.mean().reset_index().copy()
    shape_grades.insert(1, 'count', grouped.size().iloc[0])
    shape_grades.insert(2, 'true_pitch_type', 'total')    

    # concatenate
    shape_grades = pd.concat([shape_grades, pt_shape_grades], ignore_index=True)

    # rename
    shape_grades.rename(columns={col: col.replace('predicted_', '') for col in shape_grades.columns}, inplace=True)
    
    # for aesthetic purposes
    def rearrange_name(name):
        last, first = name.split(', ')[:2]
        return f"{first} {last}"

    shape_grades.insert(0, 'Name', rearrange_name(grading_data['player_name'].values[0]))
    
    return shape_grades, repertoire
query_shape_grades, query_shape_rep = grade_shape(classified_query_data)

def grade_spot(classified_pitch_data):
    
    league_heatmaps = np.load('Data/league_heatmaps.npy')
        
    pitch_classes = {'fastball': ['Riding Fastball', 'Fastball'],
                     'sinker': ['Sinker'],
                     'cutter': ['Cutter'],
                     'slider': ['Gyro Slider', 'Two-Plane Slider', 'Carry Slider', 'Sweeper'],
                     'curveball': ['Slurve', 'Curveball', 'Slow Curve'],
                     'changeup': ['Movement-Based Changeup', 'Velo-Based Changeup', 'Knuckleball']
                     }
    
    classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['fastball']), 0, np.nan)
    classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['sinker']), 1, classified_pitch_data['pitch_id'])
    classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['cutter']), 2, classified_pitch_data['pitch_id'])
    classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['slider']), 3, classified_pitch_data['pitch_id'])
    classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['curveball']), 4, classified_pitch_data['pitch_id'])
    classified_pitch_data['pitch_id'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['changeup']), 5, classified_pitch_data['pitch_id'])
    classified_pitch_data = classified_pitch_data.dropna(subset = 'pitch_id').copy()
    
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
    
    def loc_rv(data, league_heatmaps):
    
        df = data.copy()    
    
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
    
    def calculate_all_variances(classified_pitch_data, min_pitches=2, min_pitch_ratio=0.03):
        
        
        # Initialize DataFrame for results
        df = classified_pitch_data.copy()
        slot_variance_df = pd.DataFrame(
            index=[df['pitcher'].values[0]],
            columns=['pitcher', 'Name', 'release_variance', 'repertoire_variance', 'pbp_variance']
        )
        slot_variance_df['pitcher'] = df['pitcher'].values[0]
    
        
        # Set name
        last, first = df['player_name'].iloc[0].split(', ')[:2]
        slot_variance_df.loc[df['pitcher'].values[0], 'Name'] = f"{first} {last}"
        
        # Calculate overall release variance using combined std of horizontal and vertical angles
        release_points = df[['release_angle_h', 'release_angle_v']].to_numpy()
        total_std = np.sum(np.std(release_points, axis=0))
        slot_variance_df.loc[df['pitcher'].values[0], 'release_variance'] = total_std
        
        # Group by pitch type
        pitch_groups = df.groupby('true_pitch_type')
        pitch_type_stats = []
        
        # Calculate per-pitch type statistics
        for pitch_type, p_df in pitch_groups:
            if len(p_df)/len(df) < min_pitch_ratio:
                continue
            if len(p_df) < min_pitches:
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
            slot_variance_df.loc[df['pitcher'].values[0], 'repertoire_variance'] = repertoire_variance
            
            # Calculate pbp variance as weighted average of per-pitch type stds
            total_pitches = sum(stat['count'] for stat in pitch_type_stats)
            weighted_std = sum(stat['std'] * stat['count'] for stat in pitch_type_stats) / total_pitches
            slot_variance_df.loc[df['pitcher'].values[0], 'pbp_variance'] = weighted_std
    
        return slot_variance_df
    
    slot_variance_df = calculate_all_variances(classified_pitch_data)
    
    slot_variance_df[['release_variance', 'repertoire_variance', 'pbp_variance']] = slot_variance_df[['release_variance', 'repertoire_variance', 'pbp_variance']].astype(float)
    slot_variance_df = slot_variance_df.reset_index()
    
    slot_data = pd.merge(classified_pitch_data, slot_variance_df, on='pitcher', how='left')
    slot_data = slot_data.dropna(subset = 'release_variance')
    
    
    
    '''
    ###############################################################################
    ############################### Grade Pitcher #################################
    ###############################################################################
    '''

    grade_cols = ['loc_rv', 'release_variance', 'repertoire_variance', 'pbp_variance']
    meta_cols = ['game_year', 'sf_level', 'player_name']
    agg_dict = {col: 'mean' for col in grade_cols}
    agg_dict.update({col: 'first' for col in meta_cols})
    
    #
    # Pitcher level
    #
    
    pitcher_df = (
        slot_data
        .groupby('pitcher')
        .agg(agg_dict)
        .reset_index()
    )
    pitcher_df['true_pitch_type'] = 'total'
    pitcher_df['count'] = slot_data.groupby('pitcher').size().values
    
    #
    # Pitch type level
    #
    
    pitchtype_df = (
        slot_data
        .groupby(['pitcher', 'true_pitch_type'])
        .agg(agg_dict)
        .reset_index()
    )
    pitchtype_df['count'] = slot_data.groupby(['pitcher', 'true_pitch_type']).size().values
    
    loc_grades = pd.concat([pitcher_df, pitchtype_df], ignore_index=True)
    
    mean_loc = joblib.load("Data/mean_loc")
    loc_grades['loc_rv'] = (-(loc_grades['loc_rv'] - mean_loc)).round(6)
    loc_grades['loc_rv_tot'] = loc_grades['loc_rv'] * loc_grades['count']
    
    mu = 0
    var = 1
    s = 63**2
    
    n = loc_grades['count']
    x = loc_grades['loc_rv']
    
    loc_grades['proj_loc'] = ((mu/var + n*x/s) / (1/var + n/s)).round(6)
    loc_grades['proj_var'] = (1 / (1/var + n/s)).round(6)
    
    spot_grades = loc_grades[[
        'game_year', 'sf_level', 'pitcher', 'player_name',
        'true_pitch_type', 'count', 'loc_rv', 'loc_rv_tot',
        'proj_loc', 'release_variance', 'repertoire_variance', 'pbp_variance'
    ]]

    return spot_grades
query_spot_grades = grade_spot(classified_query_data)

def grade_slot(classified_pitch_data):
    
    '''
    ###############################################################################
    ############################### Slot Deviation ################################
    ###############################################################################
    '''
    
    # make clusters
    clustered_data = classified_pitch_data.dropna(subset='arm_angle').copy()
    if len(clustered_data) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    clustered_data['slot_cluster'] = (5 * round(clustered_data['arm_angle']/5)).astype(int)
    clustered_data['velo_cluster'] = (3 * round(clustered_data['release_speed']/3)).astype(int)

    pitch_classes = {'fastball': ['Riding Fastball', 'Fastball'],
                     'sinker':   ['Sinker'],
                     'breaking': ['Cutter', 'Gyro Slider', 'Two-Plane Slider', 'Carry Slider',  'Sweeper', 'Slurve', 'Curveball', 'Slow Curve'],
                     'offspeed': ['Movement-Based Changeup', 'Velo-Based Changeup', 'Knuckleball']}

    clustered_data['pitch_class'] = np.where(clustered_data['true_pitch_type'].isin(pitch_classes['fastball']), 'fastball', pd.NA)
    clustered_data['pitch_class'] = np.where(clustered_data['true_pitch_type'].isin(pitch_classes['sinker']), 'sinker', clustered_data['pitch_class'])
    clustered_data['pitch_class'] = np.where(clustered_data['true_pitch_type'].isin(pitch_classes['breaking']), 'breaking', clustered_data['pitch_class'])
    clustered_data['pitch_class'] = np.where(clustered_data['true_pitch_type'].isin(pitch_classes['offspeed']), 'offspeed', clustered_data['pitch_class'])
    clustered_data['pitch_class'] = clustered_data['pitch_class'].astype('category')

    # use GAM models to find zeroes
    slot_models = joblib.load('Data/slot_models')
    x_zero_model = slot_models[0]
    z_zero_model = slot_models[1]
    pitch_class_cats = slot_models[2]
    
    x_cols = ['slot_cluster', 'velo_cluster', 'pitch_class']
    
    # re-create categorical codes from training
    clustered_data['pitch_class'] = (
        pd.Categorical(clustered_data['pitch_class'], categories=pitch_class_cats)
    )
    X = clustered_data[x_cols].copy()
    X['pitch_class'] = X['pitch_class'].cat.codes
    X_mat = X.values
    
    # predictions
    clustered_data['x_zero'] = x_zero_model.predict(X_mat)
    clustered_data['z_zero'] = z_zero_model.predict(X_mat)

    rel_data = clustered_data.copy()
    rel_data['x_rel'] = rel_data['pfx_x'] - rel_data['x_zero']
    rel_data['z_rel'] = rel_data['pfx_z'] - rel_data['z_zero']

    eff_dict = joblib.load('Data/eff_dict')



    '''
    ###############################################################################
    ################################# Slot Rarity #################################
    ###############################################################################
    '''

    slot_frequencies = joblib.load('Data/slot_frequencies')
    rel_data['slot_rarity'] = rel_data['slot_cluster'].map(slot_frequencies)



    '''
    ###############################################################################
    ################################ Grade Pitcher ################################
    ###############################################################################
    '''
    
    def calculate_diff_rv(rel_data, valid_clusters, eff_dict):
        
        result_data = rel_data.copy()
        result_data['diff_rv'] = 0.0
        
        # Process each cluster in parallel
        for cluster in valid_clusters:
    
            velo = cluster[0]
            pitch_class = cluster[1]        
    
            # Get relevant data for this cluster
            eff_df = eff_dict[velo][pitch_class]
            
            mask = (result_data['pitch_class'] == pitch_class) & (result_data['velo_cluster'] == velo)
            pitch_data = result_data[mask]
            
            if len(pitch_data) == 0:
                continue
            
            # Create bins once for this pitch type
            xbins = eff_df.sort_values('x_bin')['pfx_x_min'].unique()
            zbins = eff_df.sort_values('z_bin')['pfx_z_min'].unique()
            
            # Vectorized bin calculation
            xbin = np.digitize(pitch_data['x_rel'].values, xbins) - 1
            zbin = np.digitize(pitch_data['z_rel'].values, zbins) - 1
            
            # Create a lookup table from eff_map
            lookup_dict = eff_df.set_index(['x_bin', 'z_bin'])['mean'].to_dict()
            
            # Vectorized lookup using list comprehension with fallback to np.nan
            rv_values = np.array([lookup_dict.get((x, z), np.nan) for x, z in zip(xbin, zbin)])
            result_data.loc[mask, 'diff_rv'] = rv_values
        
        return result_data
    
    valid_clusters = joblib.load('Data/valid_clusters')
    grade_data = calculate_diff_rv(rel_data, valid_clusters, eff_dict)
    
    min_rarity = joblib.load('Data/min_rarity')
    grade_data['diff_rv'] = grade_data['diff_rv'] * (grade_data['slot_rarity'] + min_rarity)/2 / min_rarity
    
    # aggregate
    grade_cols = ['diff_rv', 'slot_rarity', 'arm_angle']
    meta_cols = ['player_name']
    agg_dict = {col: 'first' for col in meta_cols}
    agg_dict.update({col: 'mean' for col in grade_cols})
    
    slot_grades = grade_data.groupby('pitcher').agg(agg_dict).dropna().reset_index()
    slot_grades.insert(2, 'true_pitch_type', 'total')
    pt_slot_grades = grade_data.groupby(['pitcher', 'true_pitch_type']).agg(agg_dict).dropna().reset_index()
    
    slot_grades = pd.concat([slot_grades, pt_slot_grades], ignore_index=True)
    
    # for visualization
    pbp_cols = ['release_speed', 'x_zero', 'z_zero', 'x_rel', 'z_rel']
    agg_dict.update({col: ['mean', 'std'] for col in pbp_cols})
    agg_dict.update({'slot_cluster': 'count', 'p_throws': 'first'})
    slot_info = grade_data.groupby(['pitcher', 'true_pitch_type', 'pitch_class'], observed=True).agg(agg_dict).dropna().reset_index()
    slot_info = slot_info.rename(columns = {'slot_cluster': 'count'})
    
    return slot_grades, slot_info
query_slot_grades, query_slot_info = grade_slot(classified_query_data)

def grade_sequence(classified_pitch_data):
        
    pitch_classes = {'fastball': ['Riding Fastball', 'Fastball'],
                     'sinker':   ['Sinker'],
                     'breaking': ['Cutter', 'Gyro Slider', 'Two-Plane Slider', 'Carry Slider', 
                                  'Sweeper', 'Slurve', 'Curveball', 'Slow Curve'],
                     'offspeed': ['Movement-Based Changeup', 'Velo-Based Changeup',
                                  'Knuckleball']}
    
    classified_pitch_data['pitch_class'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['fastball']), 'fastball', pd.NA)
    classified_pitch_data['pitch_class'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['sinker']), 'sinker', classified_pitch_data['pitch_class'])
    classified_pitch_data['pitch_class'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['breaking']), 'breaking', classified_pitch_data['pitch_class'])
    classified_pitch_data['pitch_class'] = np.where(classified_pitch_data['true_pitch_type'].isin(pitch_classes['offspeed']), 'offspeed', classified_pitch_data['pitch_class'])
    
    # add qaulities of previous pitch
    cols = ['pitch_number', 'at_bat_number', 'pitch_class', 'release_speed', 'plate_x', 'plate_z', 'release_angle_h', 'release_angle_v']
    prev_cols = ['prev_n', 'prev_ab', 'prev_pitch', 'prev_velo', 'prev_x', 'prev_z', 'prev_h', 'prev_v']
    classified_pitch_data[prev_cols] = pd.NA
    
    locs = classified_pitch_data.loc[:len(classified_pitch_data) - 2, cols]
    classified_pitch_data.loc[1:, prev_cols] = locs.values
    classified_pitch_data = classified_pitch_data.iloc[1:, :]
    
    # remove first pitch of PA
    classified_pitch_data.loc[classified_pitch_data['pitch_number'] == 1, prev_cols] = np.nan
    classified_pitch_data.loc[classified_pitch_data['at_bat_number'] != classified_pitch_data['prev_ab'], prev_cols] = np.nan
    classified_pitch_data = classified_pitch_data.dropna(subset=prev_cols)
    
    
    
    '''
    ###############################################################################
    ############################# Pitch Type Sequence #############################
    ###############################################################################
    '''
    
    # frequency of each outcome by previous pitch
    outcomes = ['foul', 'weak', 'topped', 'under', 'flr_brn', 'solid', 'barrel', 'swstr', 'called_strike', 'ball']
    sequence_matrices = joblib.load('Data/sequence_matrices')
    
    # run value of each outcome
    rv = joblib.load('Data/sequence_rvs')
              
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
    classified_pitch_data['sequence_rv'] = list(map(
        lambda x: sequence_dict.get((x[0], x[1]), np.nan),
        zip(classified_pitch_data['prev_pitch'], classified_pitch_data['pitch_class'])
    ))
    
    grade_cols = ['sequence_rv']
    meta_cols = ['player_name']
    agg_dict = {col: 'first' for col in meta_cols}
    agg_dict.update({col: 'mean' for col in grade_cols})
    
    sequence_grades_1 = classified_pitch_data.groupby('pitcher').agg(agg_dict).reset_index()
    sequence_grades_1.insert(2, 'true_pitch_type', 'total')
    sequence_grades_1_pt = classified_pitch_data.groupby(['pitcher', 'true_pitch_type']).agg(agg_dict).reset_index()
    
    sequence_grades_1 = pd.concat([sequence_grades_1, sequence_grades_1_pt])
    
    # quick Bayesian regression
    counts = classified_pitch_data.groupby('pitcher').size().reset_index(name='count')
    sequence_grades_1 = sequence_grades_1.merge(counts, on=['pitcher'])
    
    prior_mean = 0
    prior_weight = 100
    sequence_grades_1['sequence_rv'] = (
        sequence_grades_1['sequence_rv'] * sequence_grades_1['count'] + prior_mean * prior_weight
        ) / (sequence_grades_1['count'] + prior_weight)
    sequence_grades_1 = sequence_grades_1.drop(columns='count')
    
    
    '''
    ###############################################################################
    ############################### Micro Sequences ###############################
    ###############################################################################
    '''
    
    # define parameters
    RA_small_radius = 0.65  # degrees
    RA_large_radius = 1.45  # degrees
    loc_small_radius = 0.35 # feet
    loc_large_radius = 1.80 # feet
    
    same_release = ((classified_pitch_data['release_angle_h'] - classified_pitch_data['prev_h'])**2 + (classified_pitch_data['release_angle_v'] - classified_pitch_data['prev_v'])**2 < RA_small_radius**2)
    diff_release = ((classified_pitch_data['release_angle_h'] - classified_pitch_data['prev_h'])**2 + (classified_pitch_data['release_angle_v'] - classified_pitch_data['prev_v'])**2 > RA_large_radius**2)
    same_loc = ((classified_pitch_data['plate_x'] - classified_pitch_data['prev_x'])**2 + (classified_pitch_data['plate_z'] - classified_pitch_data['prev_z'])**2 < loc_small_radius**2)
    diff_loc = ((classified_pitch_data['plate_x'] - classified_pitch_data['prev_x'])**2 + (classified_pitch_data['plate_z'] - classified_pitch_data['prev_z'])**2 > loc_large_radius**2)
    
    # our three executable sequences
    classified_pitch_data['match'] = np.where((same_release) & (same_loc), 1, 0)
    classified_pitch_data['split'] = np.where((same_release) & (diff_loc), 1, 0)
    classified_pitch_data['freeze'] = np.where((diff_release) & (same_loc), 1, 0)
        
    grade_cols = ['match', 'split', 'freeze']
    meta_cols = ['player_name']
    agg_dict = {col: 'first' for col in meta_cols}
    agg_dict.update({col: 'mean' for col in grade_cols})
    
    sequence_grades_2 = classified_pitch_data.groupby('pitcher').agg(agg_dict).reset_index()
    sequence_grades_2.insert(2, 'true_pitch_type', 'total')
    sequence_grades_2_pt = classified_pitch_data.groupby(['pitcher', 'true_pitch_type']).agg(agg_dict).reset_index()

    sequence_grades_2 = pd.concat([sequence_grades_2, sequence_grades_2_pt])
    sequence_grades_2 = sequence_grades_2.merge(counts, on=['pitcher'])

    execution_means = joblib.load('Data/execution_means')
    for col in ['match', 'split', 'freeze']:
        prior_mean = execution_means[col]
        prior_weight = 100
        sequence_grades_2[col] = (
            sequence_grades_2[col] * sequence_grades_2['count'] + prior_mean * prior_weight
            ) / (sequence_grades_2['count'] + prior_weight)
    sequence_grades_2 = sequence_grades_2.drop(columns='count')
    
    sequence_grades = pd.merge(sequence_grades_1, sequence_grades_2, on = ['pitcher', 'player_name', 'true_pitch_type'], how='left')
    
    return sequence_grades
query_sequence_grades = grade_sequence(classified_query_data)

def predict_performance(shape_grades, repertoire, spot_grades, slot_grades, sequence_grades, results_data):
    
    KEYS = ['pitcher', 'true_pitch_type']
    
    no_slot = False
    if len(slot_grades) != len(spot_grades):
        if len(slot_grades) == 0:
            slot_grades = spot_grades.copy()
            slot_grades['diff_rv'] = 0
            slot_grades['slot_rarity'] = 0.114
            no_slot = True
        else:
            slot_grades = slot_grades.merge(spot_grades[KEYS], on=KEYS, how='right')
            slot_grades['diff_rv'] = slot_grades['diff_rv'].fillna(slot_grades.loc[0, 'diff_rv'])
            slot_grades['slot_rarity'] = slot_grades['slot_rarity'].fillna(slot_grades.loc[0, 'slot_rarity'])
    
    # combine grades
    full_df = shape_grades.copy()
    for other in [spot_grades, slot_grades, sequence_grades]:
        cols_to_add = [c for c in other.columns if c not in full_df.columns or c in KEYS]
        full_df = full_df.merge(other[cols_to_add], on=KEYS, how='outer')
        
    # combine with results
    full_df['Name'] = full_df['Name'].apply(unidecode)
    full_df['Season'] = full_df['game_year']
    if len(results_data) > 0 :
        results_df = full_df.merge(results_data, on='pitcher')
    else:
        results_df = full_df.copy()
        results_df['starter'] = 1
        results_df[['K%', 'BB%', 'SIERA']] = np.nan

    
    '''
    ###############################################################################
    ############################# Predict Performance #############################
    ###############################################################################
    '''
    
    def normalize(data, desired_mean, desired_std, pop_mean=None, pop_std=None, dtype='Int64'):
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        if pop_mean is not None:
            if isinstance(pop_mean, pd.Series):
                m = pop_mean[data.columns]
                s = pop_std[data.columns]
            else:
                m = pop_mean
                s = pop_std
        else:
            m = np.mean(data)
            s = np.std(data)
        
        if dtype == 'Int64':
            normalized_data = (((data - m) / s) * desired_std + desired_mean).round().fillna(data).astype(dtype)
        else:
            normalized_data = (((data - m) / s) * desired_std + desired_mean).astype(dtype)
        
        return normalized_data

    # describe preformance...
    def describe(data, ycol):
        
        df = data.copy()
        
        shape_cols = ['swstr', 'foul', 'weak', 'topped', 'under', 'flr_brn', 'solid', 'barrel']
        spot_cols = ['loc_rv_tot', 'proj_loc', 'pbp_variance']
        slot_cols = ['diff_rv', 'slot_rarity']
        sequence_cols = ['release_variance', 'repertoire_variance', 'sequence_rv']
        xcols = shape_cols + spot_cols + slot_cols + sequence_cols
        x_norms = joblib.load('Data/x_norms_desc')
        x_means = x_norms['mean']
        x_stds = x_norms['std']
        df[xcols] = normalize(df[xcols], 0, 1, pop_mean=x_means, pop_std=x_stds, dtype=float)
        
        results = joblib.load('Data/' + ycol + '_description_weights')
        
        sig_cols = list(results['coefficients'].keys())
        pred_vals = results['intercept'] + sum(df[col] * results['coefficients'][col] for col in sig_cols)
        
        if ycol == 'SIERA':
            spot_norms = joblib.load('Data/spot_norms')
            spot_mean = spot_norms['mean']
            spot_std = spot_norms['std']
            data['Spot+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in spot_cols), 100, 10, pop_mean=spot_mean, pop_std=spot_std)
        
        return pred_vals
        
    # and predict (spot goes in timeout)
    def predict(data, ycol):
        
        df = data.copy()
        
        shape_cols = ['swstr', 'foul', 'weak', 'topped', 'under', 'flr_brn', 'solid', 'barrel']
        spot_cols = []
        slot_cols = ['diff_rv', 'slot_rarity']
        sequence_cols = ['match', 'split', 'freeze']
        if ycol == 'BB%':
            spot_cols += ['loc_rv_tot', 'proj_loc', 'pbp_variance']
            sequence_cols += ['release_variance', 'repertoire_variance']
            
        xcols = shape_cols + spot_cols + slot_cols + sequence_cols
        x_norms = joblib.load('Data/x_norms_pred')
        if ycol == 'BB%':
            x_norms = joblib.load('Data/x_norms_pred_bb')
        x_means = x_norms['mean']
        x_stds = x_norms['std']
        df[xcols] = normalize(df[xcols], 0, 1, pop_mean=x_means, pop_std=x_stds, dtype=float)
        
        results = joblib.load('Data/' + ycol + '_prediction_weights')
        
        sig_cols = list(results['coefficients'].keys())        
        pred_vals = results['intercept'] + sum(df[col] * results['coefficients'][col] for col in sig_cols)

        if ycol == 'SIERA':
            shape_norms = joblib.load('Data/shape_norms')
            shape_mean = shape_norms['mean']
            shape_std = shape_norms['std']
            slot_norms = joblib.load('Data/slot_norms')
            slot_mean = slot_norms['mean']
            slot_std = slot_norms['std']
            sequence_norms = joblib.load('Data/sequence_norms')
            sequence_mean = sequence_norms['mean']
            sequence_std = sequence_norms['std']
            
            data['Shape+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in shape_cols), 100, 10, pop_mean=shape_mean, pop_std=shape_std)
            data['Slot+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in slot_cols), 100, 10, pop_mean=slot_mean, pop_std=slot_std)
            data['Sequence+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in sequence_cols), 100, 10, pop_mean=sequence_mean, pop_std=sequence_std)
        
        return pred_vals   
    
    for ycol in ['SIERA', 'K%', 'BB%']:
        results_df['deserved_' + ycol] = describe(results_df, ycol)

    for ycol in ['SIERA', 'K%', 'BB%']:
        results_df['predicted_' + ycol] = predict(results_df, ycol)

    # Shape+ only by handedness
    def get_shape(repertoire):
        
        df = repertoire.copy()
        
        xcols = ['swstr', 'foul', 'weak', 'topped', 'under', 'flr_brn', 'solid', 'barrel']
            
        x_norms = joblib.load('Data/x_norms_pred')
        x_means = x_norms['mean']
        x_stds = x_norms['std']
        df[xcols] = normalize(df[xcols], 0, 1, pop_mean=x_means, pop_std=x_stds, dtype=float)
        
        results = joblib.load('Data/SIERA_prediction_weights')
        
        shape_norms = joblib.load('Data/shape_norms')
        shape_mean = shape_norms['mean']
        shape_std = shape_norms['std']
            
        repertoire['Shape+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in xcols), 100, 10, pop_mean=shape_mean, pop_std=shape_std, dtype=float)
        repertoire['Shape+'] = np.round((100 * 10 + repertoire['Shape+'] * repertoire['count']) / (10 + repertoire['count'])).astype(int)

        return repertoire
    
    graded_repertoire = get_shape(repertoire)

    # finalize grades
    siera_norms = joblib.load('Data/pred_SIERA_norms')
    siera_mean = siera_norms['mean']
    siera_std = siera_norms['std']
    results_df['4S+'] = normalize(-results_df['deserved_SIERA'], desired_mean=100, desired_std=10, pop_mean=siera_mean, pop_std=siera_std, dtype='Int64')

    # pitch class norms for scouting grades
    pitch_classes = {'fastball': ['Riding Fastball', 'Fastball'],
                     'sinker':   ['Sinker'],
                     'breaking': ['Cutter', 'Gyro Slider', 'Two-Plane Slider', 'Carry Slider', 
                                  'Sweeper', 'Slurve', 'Curveball', 'Slow Curve'],
                     'offspeed': ['Movement-Based Changeup', 'Velo-Based Changeup',
                                  'Knuckleball']}
    
    pitch_class_norms = joblib.load('Data/pitch_class_norms')
    results_df['Sct. Grade'] = pd.Series([pd.NA]*len(results_df), dtype='Int64')

    for pitch_class in pitch_class_norms.keys():
        mask = results_df['true_pitch_type'].isin(pitch_classes[pitch_class])
        if mask.sum() == 0:
            continue
    
        pop_mean = pitch_class_norms[pitch_class]['mean']
        pop_std = pitch_class_norms[pitch_class]['std']
        
        normed = normalize(-results_df.loc[mask, 'predicted_SIERA'],
                       desired_mean=50, desired_std=10,
                       pop_mean=pop_mean, pop_std=pop_std)
    
        results_df.loc[mask, 'Sct. Grade'] = normed.to_numpy()

    cols_of_interest = ['Name', '4S+', 'pitcher', 'starter', 'true_pitch_type', 
                        'count', 'Sct. Grade', 'release_speed', 'pfx_x', 'pfx_z',
                        'Shape+', 'Spot+', 'Slot+', 'Sequence+', 
                        'K%', 'deserved_K%', 'predicted_K%',
                        'BB%', 'deserved_BB%', 'predicted_BB%',
                        'SIERA', 'deserved_SIERA', 'predicted_SIERA']
    grades = results_df[cols_of_interest]
    grades = grades.sort_values(by = 'predicted_SIERA', ascending = True).reset_index(drop=True)
    grades['SIERA_diff'] = grades['SIERA'] - grades['predicted_SIERA']

    grades[['SIERA', 'deserved_SIERA', 'predicted_SIERA']] = round(grades[['SIERA', 'deserved_SIERA', 'predicted_SIERA']], 2)
    grades[['K%', 'BB%', 'deserved_K%','deserved_BB%', 'predicted_K%', 'predicted_BB%']] = round(100*grades[['K%', 'BB%', 'deserved_K%','deserved_BB%', 'predicted_K%', 'predicted_BB%']], 1)
    grades = grades.sort_values(by='count', ascending=False).reset_index(drop=True)

    return grades, graded_repertoire, no_slot
query_grades, query_repertoire, no_slot = predict_performance(query_shape_grades, query_shape_rep, query_spot_grades, query_slot_grades, query_sequence_grades, results_data)

def profile_viz(pitcher_grades, shape_by_hand, classified_pitch_data, conditions, no_slot, display_mode='All', quality='high'):
    
    '''
    ###############################################################################
    ############################# Function Definition #############################
    ###############################################################################
    '''
    
    def get_data(classified_pitch_data, pitcher_grades):
        
        df = pitcher_grades.iloc[0].copy()
        if len(df) == 0:
            print('\nGrades could not be compiled for this pitcher.')
            return ['fail', 'fail', 'fail', 'fail']
            
        pitcher_data = {
            "Name": df['Name'],
            "SP": int(df['starter']),
            "pitcher_sfg_id": df['pitcher'],
            "Pitches Thrown": df['count'],
            "4S+": int(round(df['4S+'])),
        }
        
        hand = classified_pitch_data['p_throws'].values[0]
        pitcher_data['Hand'] = hand
        
        shape = df['Shape+']
        spot = df['Spot+']
        slot = df['Slot+']
        sequence = df['Sequence+']
    
        shape_coeff, spot_coeff, slot_coeff, sequence_coeff = joblib.load('Data/broad_coeffs')
        shape_rv = shape_coeff * (shape - 100)/10 * (5/6) / 100
        spot_rv = spot_coeff * (spot - 100)/10 * (5/6) / 100
        slot_rv = slot_coeff * (slot - 100)/10 * (5/6) / 100
        sequence_rv = sequence_coeff * (sequence - 100)/10 * (5/6) /100
        total_rv = shape_rv + spot_rv + slot_rv + sequence_rv
            
        grades = {
            "Shape": int(round(shape)),
            "Slot": int(round(slot)),
            "Sequence": int(round(sequence)),
            "Spot": int(round(spot)),
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
            "K%": [f'{df["K%"]:.1f}', f'{df["deserved_K%"]:.1f}', f'{df["predicted_K%"]:.1f}'],
            "BB%": [f'{df["BB%"]:.1f}', f'{df["deserved_BB%"]:.1f}', f'{df["predicted_BB%"]:.1f}'],
            "SIERA": [f'{df["SIERA"]:.2f}', f'{df["deserved_SIERA"]:.2f}', f'{df["predicted_SIERA"]:.2f}'],
        })
        
        return pitcher_data, grades, xrv, performance_data
    
    def grade_repertoire(pitcher, pitcher_grades, shape_by_hand, classified_pitch_data, display_mode='Shape'):
        
        hand = classified_pitch_data['p_throws'].values[0]
        
        if display_mode == 'Shape':
            
            df = shape_by_hand.copy()
            
            if hand == 'R':
                map_R = 0
                map_L = 1
            else:  
                map_R = 1
                map_L = 0
        
            df_R = df[df['platoon'] == map_R][['true_pitch_type', 'percent', 'Shape+']].copy()
            df_L = df[df['platoon'] == map_L][['true_pitch_type', 'percent', 'Shape+']].copy()
        
            df_R = df_R.rename(columns={
                'percent': 'R',
                'Shape+': 'Shape+ R'
            })
        
            df_L = df_L.rename(columns={
                'percent': 'L',
                'Shape+': 'Shape+ L'
            })
        
            # Merge into one row per pitch type
            repertoire = (
                df_R
                .merge(df_L, on='true_pitch_type', how='outer')
                .sort_values('R', ascending=False)
                .reset_index(drop=True)
                .fillna(0)
            )
            
            try:
                repertoire.drop(repertoire[repertoire['R'] + repertoire['L'] < 1].index[0], inplace=True)
            except IndexError:
                pass
            
            repertoire[['Shape+ R', 'Shape+ L', 'R', 'L']] = round(repertoire[['Shape+ R', 'Shape+ L', 'R', 'L']]).astype(int).astype(str)
            repertoire.loc[repertoire['R'] > '0', 'R'] += '%'
            repertoire.loc[repertoire['L'] > '0', 'L'] += '%'
            repertoire[['Shape+ R', 'Shape+ L', 'R', 'L']] = repertoire[['Shape+ R', 'Shape+ L', 'R', 'L']].replace({'0': '--'})    
        
            repertoire.loc[repertoire['R'] == '--', 'Shape+ R'] = '--'
            repertoire.loc[repertoire['L'] == '--', 'Shape+ L'] = '--'
            
        elif display_mode == 'Scouting':
            repertoire = pitcher_grades.iloc[1:].copy()
            repertoire = repertoire[repertoire['count'] > 0.01 * repertoire['count'].sum()]
            repertoire['Sct. Grade'] = np.clip(repertoire['Sct. Grade'], 20, 80)
            repertoire[['pfx_x', 'pfx_z']] = round(repertoire[['pfx_x', 'pfx_z']]).astype(int)
            repertoire[['release_speed']] = round(repertoire[['release_speed']], 1)
            repertoire = repertoire[['true_pitch_type', 'release_speed', 'pfx_z', 'pfx_x', 'Sct. Grade']]
        
        elif display_mode == 'All':
            
            df = shape_by_hand.copy()
            
            if hand == 'R':
                map_R, map_L = 0, 1
            else:
                map_R, map_L = 1, 0
            
            # --- handedness splits ---
            df_R = (
                df[df['platoon'] == map_R]
                [['true_pitch_type', 'percent', 'Shape+']]
                .rename(columns={
                    'percent': 'R',
                    'Shape+': 'Shape+ R'
                })
            )
            
            df_L = (
                df[df['platoon'] == map_L]
                [['true_pitch_type', 'percent', 'Shape+']]
                .rename(columns={
                    'percent': 'L',
                    'Shape+': 'Shape+ L'
                })
            )
            
            # --- merge into one row per pitch ---
            repertoire = (
                df_R
                .merge(df_L, on='true_pitch_type', how='outer')
                .fillna(0)
            )
            
            # drop pitches with negligible total usage
            repertoire = repertoire[repertoire['R'] + repertoire['L'] >= 1]
            
            # --- merge pitch characteristics ---
            repertoire_2 = pitcher_grades.iloc[1:].copy()
            repertoire_2 = repertoire_2[
                repertoire_2['count'] > 0.01 * repertoire_2['count'].sum()
            ]
            
            repertoire_2['Sct. Grade'] = np.clip(repertoire_2['Sct. Grade'], 20, 80)
            repertoire_2['pfx_x'] = repertoire_2['pfx_x'].round(0)
            repertoire_2['pfx_z'] = repertoire_2['pfx_z'].round(0)
            repertoire_2['release_speed'] = repertoire_2['release_speed'].round(1)
            
            repertoire_2 = repertoire_2[[
                'true_pitch_type',
                'release_speed',
                'pfx_z',
                'pfx_x',
                'Sct. Grade',
                'count'
            ]]
            
            # --- final merge ---
            repertoire = (
                repertoire
                .merge(repertoire_2, on='true_pitch_type', how='inner')
                .sort_values('count', ascending=False)
                .reset_index(drop=True)
            )

        else:
            print('\nThe only available display modes are "Shape" and "Scouting"')
            return

        repertoire.rename(columns={'true_pitch_type': 'Pitch Type', 
                                   'release_speed': 'Velo',
                                   'pfx_x': 'HB',
                                   'pfx_z': 'iVB',
                                   'count': 'usage'}, inplace=True)

        if 'Movement-Based Changeup' in repertoire['Pitch Type'].values:
            if len(repertoire) < 2:
                repertoire.loc[repertoire['Pitch Type'] == 'Movement-Based Changeup', 'Pitch Type'] = 'Movement-Based\n Changeup'
            else:
                repertoire.loc[repertoire['Pitch Type'] == 'Movement-Based Changeup', 'Pitch Type'] = 'M-B Change'
        if 'Velo-Based Changeup' in repertoire['Pitch Type'].values:
            if len(repertoire) < 2:
                repertoire.loc[repertoire['Pitch Type'] == 'Velo-Based Changeup', 'Pitch Type'] = 'Velo-Based\n Changeup'
            else:
                repertoire.loc[repertoire['Pitch Type'] == 'Velo-Based Changeup', 'Pitch Type'] = 'V-B Change'
    
        return repertoire
    
    def plot_repertoire(pitcher, classified_pitch_data):
             
        # get all pitches
        df = classified_pitch_data.copy()
        
        # pitcher handedness
        hand = df['p_throws'].values[0]
        
        # pitches + corresponding colors
        pitch_names = np.array([
            'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider', 'Carry Slider', 
            'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup', 
            'Velo-Based Changeup', 'Knuckleball'
        ])
        
        colors = np.array(['red', 'tomato', 'darkorange', 'sienna', 'xkcd:sunflower yellow', 
                            'yellowgreen', 'green', 'limegreen', 'lightseagreen', 
                            'cornflowerblue', 'mediumpurple', 'darkgoldenrod',
                            'goldenrod', 'gray'])
        
        colordict = dict(zip(pitch_names, colors))
    
        # add colors
        df['color'] = df['true_pitch_type'].map(colordict)
        df = df.dropna(subset = 'color')
    
        # redo this for plotting
        pitch_names = np.array([
            'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider', 'Carry Slider', 
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


    ipython = get_ipython()
    ipython.run_line_magic('matplotlib', 'qt')    

    pitcher_data, grades, xrv, performance_data = get_data(classified_pitch_data, pitcher_grades)
    if pitcher_data == 'fail':
        return
    pitcher = pitcher_data['pitcher_sfg_id']
    
    # Create figure with subplots (7x6 layout)
    if quality == 'low':
        big_fig = plt.figure(figsize=(10, 10), dpi=80)
    else:
        big_fig = plt.figure(figsize=(10, 10), dpi=100)
    
    gs = big_fig.add_gridspec(7, 6)
    ax_title = big_fig.add_subplot(gs[0, :])
    
    ax_polar = big_fig.add_subplot(gs[1:4, :3], polar=True)
    ax_rv = big_fig.add_subplot(gs[1, 3:])
    ax_perf = big_fig.add_subplot(gs[2:3, 3:])
    
    if display_mode == 'Breakdown':
        ax_brk = big_fig.add_subplot(gs[4:, :])
    else:
        ax_viz = big_fig.add_subplot(gs[4:, 3:])
        ax_rep = big_fig.add_subplot(gs[4:, :3])
    ax_cred = big_fig.add_subplot(gs[6, :])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    
    
    # Top stuff    
    name = pitcher_data['Name']
    hand = pitcher_data['Hand'] + 'H'
    if pitcher_data['SP'] == 1:
        pos_string = 'SP'
    else:
        pos_string = 'RP'
    
    # display query conditions
    sd = conditions['start_dt']
    ed = conditions['end_dt']
    lvl = classified_pitch_data['sf_level'].unique()
    gt = classified_pitch_data['game_type'].unique()
    
    # dates
    months = {'01': 'January',
              '02': 'February',
              '03': 'March',
              '04': 'April',
              '05': 'May',
              '06': 'June',
              '07': 'July',
              '08': 'August',
              '09': 'September',
              '10': 'October',
              '11': 'November',
              '12': 'December'}
    
    if sd is None or ed is None:
        earliest_pitch = classified_pitch_data.sort_values(by = ['game_year', 'month', 'day'])
        sy = earliest_pitch['game_year'].values[0]
        ey = earliest_pitch['game_year'].values[-1]
    
        if sy == ey:
            date_cond = str(sy) + ' Season'
        else:
            date_cond= str(sy) + '-' + str(ey) + ' Seasons'
        
    else:
        if sd == ed:
            date_cond = months[sd[5:7]] + ' ' + str(int(sd[8:])) + ', ' + sd[:4]
        
        elif int(sd[5:7]) < 3 and int(ed[5:7]) > 10:
            if int(sd[:4]) == int(ed[:4]):
                date_cond = sd[:4] + ' Season'
            else:
                date_cond = sd[:4] + '-' + ed[:4] + ' Seasons'
        
        else:
            if int(sd[8:]) == 1 and int(ed[8:]) > 29:
                if int(sd[:4]) == int(ed[:4]):
                    if int(sd[5:7]) == int(ed[5:7]):
                        date_cond = months[sd[5:7]] + ' ' + sd[:4]
                    else:
                        date_cond = months[sd[5:7]] + ' - ' + months[ed[5:7]] + ', ' + sd[:4]
                else:
                    date_cond = months[sd[5:7]] + ' ' + sd[:4] + ' - ' + months[ed[5:7]] + ', ' + ed[:4]
            else:
                if int(sd[:4]) == int(ed[:4]): 
                    if int(sd[5:7]) == int(ed[5:7]):
                        date_cond = months[sd[5:7]] + ' ' + str(int(sd[8:])) + '-' + str(int(ed[8:])) + ', ' + sd[:4]
                    else:
                        date_cond = months[sd[5:7]] + ' ' + str(int(sd[8:])) + ' - ' + months[ed[5:7]] + ' ' + str(int(ed[8:])) + ', ' + sd[:4]
                else:
                    date_cond = months[sd[5:7]] + ' ' + str(int(sd[8:])) + sd[:4] + ' - ' + months[ed[5:7]] + ' ' + str(int(ed[8:])) + ', ' + sd[:4]
                
    # level(s)
    lvl_dict = {'mlb': 'MLB',
                'aaa': 'AAA',
                'aax': 'AA',
                'afa': 'Hi-A',
                'afx': 'Lo-A',
                'rok': 'ROK',
                'acl': 'ROK',
                'dsl': 'ROK',
                'afl': 'AFL',
                'spr': 'MLB',
                'major_league_spring_training': 'MLB',
                'minor_league_spring_training': 'MiLB',
                'minor_league_extended_spring_training': 'MiLB',
                'world_baseball_classic': 'WBC',
                'instructional_league': 'Instructs',
                'win': 'Winter Ball'}
                
    if len(lvl) == 1:
        lvl = lvl[0]
        if lvl == 'mlb':
            try:
                teams = classified_pitch_data['pitcher_team'].unique()
                lvl_cond = ' for ' + ', '.join(teams) 
            except KeyError:
                lvl_cond = ' at MLB'
        else:
            lvl_translated = lvl_dict.get(lvl, lvl.upper())
            if '_' in lvl_translated:
                lvl_translated = ' '.join(lvl_translated.split('_'))
            lvl_cond = ' at ' + lvl_translated

    else:
        seen = set()
        seen.add('Spring Training')
        unique_lvls = []
        for k, v in lvl_dict.items():
            if k in lvl and v not in seen:
                seen.add(v)
                unique_lvls.append(v)
        lvl_cond = ' at ' + ' / '.join(unique_lvls)
    
    # game type(s)
    type_dict = {'A': 'All-Star Game',
                 'C': 'Postseason',
                 'D': 'Postseason',
                 'E': 'Exhibition Game',
                 'F': 'Postseason',
                 'L': 'Postseason',
                 'S': 'Spring Training',
                 'W': 'Postseason',
                 'X': 'Unofficial Games'}
    
    if gt is None:
        type_cond = ''
    elif len(gt) == 1:
        if gt[0] == 'R':
            type_cond = ''
        else:
            type_cond = f' ({type_dict[gt[0]]})'
    else:
        gt = list(gt)
        if 'R' in gt:
            gt.remove('R')
            unique_gts = list({type_dict[t] for t in gt})
            type_cond = ' (incl. ' + ', '.join(unique_gts) + ')'
        else:
            unique_gts = list({type_dict[t] for t in gt})
            type_cond = ' (' + ', '.join(unique_gts) + ')'
    
    ax_title.text(0.5,0.6,f"{name} ({hand} {pos_string}) - Pitching Profile", fontsize=16, ha='center', va='center', weight='bold')
    ax_title.text(0.5,0.35,f"{date_cond}{lvl_cond}{type_cond}", fontsize=14, ha='center', va='center', weight='bold')
    ax_title.axis('off')
    
    
    
    #
    # 4S Grades
    #
    
    
    
    # Radar chart 
    
    categories = list(grades.keys())
    values = np.array(list(grades.values()))
    
    # Normalize values (by percentile?)
    center_radius = 0.23
    # min_radius = 0.45
    # max_radius = 0.95
    # z_scores = (values - 100) / 10
    # percentiles = stats.norm.cdf(z_scores)
    # normalized_values = percentiles * (max_radius - min_radius) + min_radius
    normalized_values = values/200 + center_radius

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
    for bar, color_name in zip(bars, ['green', 'red', 'orange', 'blue']):
        
        if no_slot and color_name == 'red':
            continue
        
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
    
    
    
    # Add curved labels
    widths = joblib.load('Data/character_widths_16_bold')
    dpi = big_fig.dpi
    if quality == 'low':
        dpi_factor = 3.5
    else:
        dpi_factor = 3
    dpi = dpi * dpi_factor
    
    def write_curved_text(ax, text, radius, theta_mid, direction):
        
        n = len(text)
        white_space_angle = 5 / (radius * dpi)
        angles = []
        for char in text:  
            pixel_width = widths[char]
            arc = pixel_width / (radius * dpi)
            angles.append(arc)
        total_arc = np.abs(np.sum(angles)) + (n-1)*white_space_angle
        
        if direction == 'cw':
            orientation = -1
        else:
            orientation = 1
            radius += 30/dpi
            
        theta_start = theta_mid - orientation * total_arc/2
        theta = theta_start
        for i, char in enumerate(text):
            theta = theta + orientation * (angles[i]/2)
            ax.text(theta, radius, char,
                    rotation=np.rad2deg(theta + orientation*np.pi/2),
                    rotation_mode='anchor',
                    ha='center', va='bottom', 
                    fontsize=16, fontweight='bold')
            theta = theta + orientation * (angles[i]/2 + white_space_angle)
            
        
    write_curved_text(ax_polar, f'{categories[0]}+', radius=values[0]+0.05, theta_mid=angles[0], direction='cw')
    write_curved_text(ax_polar, f'{categories[1]}+', radius=values[1]+0.05, theta_mid=angles[1], direction='ccw')
    write_curved_text(ax_polar, f'{categories[2]}+', radius=values[2]+0.05, theta_mid=angles[2], direction='ccw')
    write_curved_text(ax_polar, f'{categories[3]}+', radius=values[3]+0.05, theta_mid=angles[3], direction='cw')


    # # old flat labels
    # ax_polar.text(angles[0], values[0] + 0.05, categories[0] + '+', ha='right', va='bottom', fontsize=16, fontweight='bold')
    # ax_polar.text(angles[1], values[1] + 0.05, categories[1] + '+', ha='right', va='top', fontsize=16, fontweight='bold')
    # ax_polar.text(angles[2], values[2] + 0.05, categories[2] + '+', ha='left', va='top', fontsize=16, fontweight='bold')
    # ax_polar.text(angles[3], values[3] + 0.05, categories[3] + '+', ha='left', va='bottom', fontsize=16, fontweight='bold')
    
    ax_polar.text(angles[0], (values[0] + center_radius)/2, str(list(grades.values())[0]), ha='center', va='center', fontsize=19, fontweight='bold', color='black', path_effects=[path_effects.withStroke(linewidth=1, foreground='lightgray')])
    if no_slot:
        ax_polar.text(angles[1], (values[1] + center_radius)/2, '--', ha='center', va='center', fontsize=19, fontweight='bold', color='black', path_effects=[path_effects.withStroke(linewidth=1, foreground='lightgray')])
    else:
        ax_polar.text(angles[1], (values[1] + center_radius)/2, str(list(grades.values())[1]), ha='center', va='center', fontsize=19, fontweight='bold', color='black', path_effects=[path_effects.withStroke(linewidth=1, foreground='lightgray')])
    ax_polar.text(angles[2], (values[2] + center_radius)/2, str(list(grades.values())[2]), ha='center', va='center', fontsize=20, fontweight='bold', color='black', path_effects=[path_effects.withStroke(linewidth=1, foreground='lightgray')])
    ax_polar.text(angles[3], (values[3] + center_radius)/2, str(list(grades.values())[3]), ha='center', va='center', fontsize=20, fontweight='bold', color='black', path_effects=[path_effects.withStroke(linewidth=1, foreground='lightgray')])
    
    
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
        rep_lvl = -22
    else:
        n_pitches = 1000
        rep_lvl = -4

    rvs = list(n_pitches*np.array([xrv['Shape_RV'], xrv['Spot_RV'], xrv['Slot_RV'], xrv['Sequence_RV']]))
    war = round((n_pitches * xrv['xRV'] - rep_lvl) / 10, 1)
    rvs = list((np.round(np.array(rvs)).astype(int)))
    rvs.append(war)
    
    for i in range(len(rvs)):
        if rvs[i] > 0.05:
            rvs[i] = '+' + str(rvs[i])
    
    ax_rv.text(0.5, 1, 'Extrapolated full season run value', fontsize=14, ha='center', va='center', weight='bold')
    top = 0.83
    ax_rv.text(0.5, top-0.68, f'*Full season for {pos_string} defined as {n_pitches} pitches', ha='center', va='center', fontsize=8)

    
    def get_text_width(word):
        font = ImageFont.load_default()
        bbox = font.getbbox(word)
        return bbox[2]
    
    words = ['Shape', 'Spot', 'Slot', 'Sequence', 'WAR']
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
        ax_rv.text(places[i], top-0.15, word, fontsize=12, ha='center', va='center')
        ax_rv.text(places[i], top-0.35, rvs[i], fontsize=12, ha='center', va='center', weight='bold')
        
        if i < len(words) - 1:
            
            x = places[i] + (widths[i]/2 + dx/2) * available_width / table_width
            
            ax_rv.vlines(x, top-0.5, top, color='black', lw=1)
            
    ax_rv.set_xlim(0, 1)  # Keep fixed limits for the subplot
    ax_rv.set_ylim(0, 1)
    
    
    
    #
    # Performance data table
    #
    
    ax_perf.axis('off')
    table = ax_perf.table(cellText=performance_data.values, colLabels=performance_data.columns, colWidths = [0.2, 0.2, 0.2, 0.2], loc='center', cellLoc='center', bbox = [0.1, -0.63, 0.8, 1.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    ax_perf.set_xlim(0, 1) 
    ax_perf.set_ylim(0, 1)
    ax_perf.text(0.5, 0.72, 'Performance Metrics', ha='center', va='bottom', fontsize=14, weight='bold')
    
    
    
    
    
    #
    # Grade breakdowns, if desired
    #  
    
    if display_mode == 'Breakdown':
        
        ax_brk.clear()
        ax_brk.axis('off')
        ax_brk.set_xlim((0,1))
        ax_brk.set_ylim((0,1))
        
        granular_categories = pitcher_grades.columns[-9:-1]
        ncat = len(granular_categories)
        
        # title
        ax_brk.text(0.5, 1, 'Grade Breakdown - Percentiles', fontsize=14, weight='bold', ha='center', va='top')
        
        base_colors = {
            0: '#006400',  # Dark green
            1: '#00008B',  # Dark blue
            2: '#8B0000',  # Dark red
            3: '#8B4500'   # Dark orange
        }

        light_colors = {i: mcolors.to_hex(0.2 * np.array(mcolors.to_rgb(base_colors[i])) + 0.8*np.array([1, 1, 1]))  for i in range(len(base_colors))}

        # set outlines
        extra_left = 0.05
        extra_right = 0.95
        extra_top = 0.94
        extra_bot = 0.17

        
        left = extra_left + 0.07
        right = extra_right - 0.02
        top = extra_top - 0.03
        bot = extra_bot + 0.03

        bubble_ends = [extra_left + 0.25, right-0.0125]
        bubble_len = bubble_ends[1] - bubble_ends[0]
        offset = (top - bot) / ncat
        ax_brk.vlines(x=[extra_left, extra_right, bubble_ends[0]], ymin=extra_bot, ymax=extra_top, color='k', linestyle='-', linewidth=1)
        # ax_brk.vlines(x=bubble_ends[1], ymin=extra_bot, ymax=extra_top, color='k', linestyle='--', linewidth=1.5, alpha=0.3)
        ax_brk.hlines(y=np.linspace(bot, top, (ncat+1))[1:-1], xmin=left, xmax=bubble_ends[0], color='k', linestyle=':', linewidth=1.5)
        ax_brk.hlines(y=[extra_bot, extra_top], xmin=extra_left, xmax=extra_right, color='k', linestyle='-', linewidth=1)
        
        def get_color(c, percentile):
            
            base_color = base_colors[np.floor(c/2)]
            base_rgb = mcolors.to_rgb(base_color)
            
            white = np.array([1, 1, 1])
            strength = percentile / 100 
            if strength < 0.2:
                strength = 0.2
            color = white * (1 - strength) + np.array(base_rgb) * strength
            
            return color
        
        for a, attribute in enumerate(granular_categories):
            
            percentile = pitcher_grades[attribute].values[0]
            if percentile == 100:
                percentile = 99
            
            index = int(np.floor(a/2))
            h = offset
            r = h/2 * 3/7
            lw = 0.004
            mid_y = top-a*h-h/2-lw
            xmin = bubble_ends[0]
            xmax = xmin + bubble_len*percentile/100
            
            # label
            ax_brk.text(xmin-0.01, mid_y, attribute, color=base_colors[index], ha='right', va='center', fontsize=14, fontweight='bold')
            
            # noise maker
            final_color = get_color(a, percentile)
            cmap = LinearSegmentedColormap.from_list('custom', [light_colors[index], final_color])
            dx = 0.1 * bubble_len
            n_segments = int(np.floor((xmax - xmin)/dx))
            remainder = xmax - (xmin + dx * n_segments)
            for i in range(n_segments):
                width = xmin + i * dx
                ax_brk.barh(mid_y + lw,
                            dx,
                            left=width,
                            height=h-2*lw,
                            color=cmap(i/n_segments), 
                            edgecolor=None,
                            alpha=1)
            ax_brk.barh(mid_y + lw,
                        remainder,
                        left=xmax - remainder,
                        height=h-2*lw,
                        color=cmap(0.999), 
                        edgecolor=None,
                        alpha=1)
                        
            # # bubble
            # rx = r
            # ry = h/2 - lw
            # num_pts = 50
            # theta = np.linspace(-np.pi / 2, np.pi / 2, num_pts)
            # xs = (xmax-lw) + rx * np.cos(theta)
            # ys = mid_y+lw + ry * np.sin(theta)
            # verts = [(x, y) for x, y in zip(xs, ys)]
            # verts.append((xmax, mid_y - ry))
            # codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
            # path = Path(verts, codes)
            # cap = PathPatch(path, facecolor=final_color, edgecolor=None, linewidth=0)
            # ax_brk.add_patch(cap)
            
            # bubble label
            ax_brk.text(xmax+r, mid_y, str(percentile), ha='center', va='center', fontsize=14, fontweight='bold')
            
            
        ax_cred.axis('off')
        ax_cred.text(0.5, 0.3, 'Data and visualization are property of Major League Baseball and the Chicago White Sox.', ha='center', va='center', fontsize=10)
        plt.show()
        return
        
    
    
    #
    # Repertoire viz
    #
    
    fig, ax, colordict = plot_repertoire(pitcher, classified_pitch_data)
    
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
    
    # Display the image in the target axes
    ax_viz.clear()    
    ax_viz.set_xlim((0,8))
    ax_viz.set_ylim((0,7))
    
    x1 = 0.3
    y1 = 1.4
    
    width = 8.8
    height = width/aspect_ratio
    
    x2 = x1 + width
    y2 = y1 + height
    
    ax_viz.imshow(image, extent=(x1,x2,y1,y2))
    ax_viz.text(4, 7, 'Movement Profile', fontsize=14, weight='bold', ha='center', va='center')
    ax_viz.axis('off')
    
    # Close the original figure
    plt.close(fig)

    
    
    
    
        
    #
    # Repertoire panel
    #
    
    ax_rep.axis('off')
    ax_rep.set_xlim(0, 1)
    ax_rep.set_ylim(0, 1)
    
    repertoire = grade_repertoire(
        pitcher,
        pitcher_grades,
        shape_by_hand,
        classified_pitch_data,
        display_mode=display_mode
    )
    
    n = len(repertoire)
    
    colordict['M-B Change'] = colordict['Movement-Based\n Changeup']
    colordict['V-B Change'] = colordict['Velo-Based\n Changeup']
    
    # params
    ABS_BOT = 0.14
    ABS_TOP = 0.98
    ABS_LEFT = 0.02
    ABS_RIGHT = 1.02
    
    PCT_WIDTH = 0.265
    GRADE_WIDTH = 0.04
    MID_WIDTH = 0.27
    # 2p + 2g + m = 1
    
    TITLE_BUFFER = 0.03
    COL_BUFFER = 0.02
    ROW_BUFFER = 0.016
    PCT_BUFFER = 0.01
    SQUEEZE = 0.07
    USAGE_BAR_PCT = 0.95
    
    FONT_SIZE = 11
    IDEAL_HT = 0.24
    MIN_HT = 0.10
    MAX_N = 8
    N_LIM = 3
    
    row_ht = IDEAL_HT if n <= N_LIM else IDEAL_HT - (IDEAL_HT - MIN_HT) / MAX_N * n
    
    table_ht = row_ht * n
    table_top = (ABS_TOP + ABS_BOT)/2 + table_ht/2
    table_bot = (ABS_TOP + ABS_BOT)/2 - table_ht/2
    tippy_top = table_top + TITLE_BUFFER
    
    # ---- title ----
    ax_rep.text(
        (ABS_LEFT+ABS_RIGHT)/2, tippy_top,
        'Repertoire Grades',
        fontsize=14,
        weight='bold',
        ha='center',
        va='bottom'
    )
    
    # ---- column regions ----
    x1 = ABS_LEFT + COL_BUFFER
    x2 = x1 + PCT_WIDTH
    x3 = x2 + COL_BUFFER
    x4 = x3 + GRADE_WIDTH
    x5 = x4 + COL_BUFFER
    x6 = x5 + MID_WIDTH
    x7 = x6 + COL_BUFFER
    x8 = x7 + GRADE_WIDTH
    x9 = x8 + COL_BUFFER
    x10 = x9 + PCT_WIDTH
    
    L_COL = (x1, x2)
    CL_COL = (x3, x4)
    C_COL = (x5, x6)
    CR_COL = (x7, x8)
    R_COL = (x9, x10)
    
    # ---- row geometry ----
    row_h = (table_top - table_bot) / n
    y_centers = [table_top - (i + 0.5) * row_h for i in range(n)]
    y_borders = [table_top - i * row_h for i in range(n + 1)]
    x_borders = [x2, x5, x6, x9]
    
    # ---- borders ---- (just around header + edges and grades)
    
    # vertical lines
    for i in range(len(x_borders)):
        ax_rep.plot([x_borders[i], x_borders[i]], [table_bot, table_top], linewidth=1, color='k')
    
    # horizontal lines
    for i in range(len(y_borders)):
        ax_rep.plot([x_borders[0], x_borders[-1]], [y_borders[i], y_borders[i]], linewidth=1, color='k')

    # ---- headers ----
    header_y = table_top
    
    ax_rep.text(sum(L_COL)/2, header_y, 'vs LHH', ha='center', va='bottom', fontsize=FONT_SIZE)
    ax_rep.text(sum(R_COL)/2, header_y, 'vs RHH', ha='center', va='bottom', fontsize=FONT_SIZE)

    # ---- usage scaling ----
    usage_max = repertoire[['L', 'R']].to_numpy().max()

    # ---- drawing helpers ----
    def draw_bar(ax, x0, x1, y, frac, color, height, direction="right", rounded=True, alpha=0.5, offset=0):
    
        total_width = x1 - x0
        width = total_width * frac + offset
    
        x = x0 - offset if direction == 'right' else x1 - width + offset

        boxstyle = "round,pad=0,rounding_size=0.05" if rounded else "square,pad=0"
    
        rect = mpatches.FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle=boxstyle,
            linewidth=0,
            facecolor=color,
            alpha=alpha,
        )
    
        ax.add_patch(rect)
        return width - offset

    
    # ---- rows ----
    for i, row in repertoire.iterrows():
        y = y_centers[i]
        pitch = row['Pitch Type']
        color = colordict[pitch]
    
        # -------- center column --------
        draw_bar(
            ax_rep,
            C_COL[0],
            C_COL[1],
            y,
            1,
            color,
            height=row_h,
            rounded=False,
        )    
        
        ax_rep.text(
            sum(C_COL)/2,
            y + ROW_BUFFER * (1 - 0.5*np.floor(n/5) - 0.4*np.floor(n/7)),
            pitch,
            ha='center',
            va='bottom',
            fontsize=FONT_SIZE - np.floor(n/7),
        )
    
        ax_rep.text(
            sum(C_COL)/2,
            y - ROW_BUFFER * (1 - 0.5*np.floor(n/5) - 0.4*np.floor(n/7)),
            f"{row['Velo']:.1f} | "
            f"{row['iVB']:.0f}\" | "
            f"{row['HB']:.0f}\"",
            ha='center',
            va='top',
            fontsize=FONT_SIZE - np.floor(n/7),
        )

        # -------- vs LHH --------
        if row['L'] > 1:
            frac_L = row['L'] / usage_max
            bar_w = draw_bar(
                ax_rep,
                L_COL[0],
                L_COL[1],
                y,
                frac_L,
                color,
                height=row_h*USAGE_BAR_PCT,
                direction='left',
                offset=(CL_COL[1] - CL_COL[0])
            )
            
            if bar_w > SQUEEZE + PCT_BUFFER:
                x = L_COL[1] - bar_w + PCT_BUFFER
                ha = 'left'
            else:
                x = L_COL[1] - bar_w - PCT_BUFFER
                ha = 'right'
            
            ax_rep.text(
                x,
                y,
                f"{row['L']:.0f}%",
                ha=ha,
                va='center',
                fontsize=FONT_SIZE,
            )
    
            draw_bar(
                ax_rep,
                CL_COL[1],
                L_COL[1],
                y,
                1,
                color='#FFFFFF',
                height=row_h,
                direction='left',
                rounded=False,
                alpha=1
            )
    
            ax_rep.text(
                sum(CL_COL) / 2,
                y,
                f"{row['Shape+ L']:.0f}",
                ha='center',
                va='center',
                weight='bold',
                fontsize=FONT_SIZE,
            )
    
        # -------- vs RHH --------
        if row['R'] > 1:
            frac_R = row['R'] / usage_max
            bar_w = draw_bar(
                ax_rep,
                R_COL[0],
                R_COL[1],
                y,
                frac_R,
                color,
                height=row_h*USAGE_BAR_PCT,
                offset=(CR_COL[1] - CR_COL[0])
            )
    
            if bar_w > SQUEEZE + PCT_BUFFER:
                x = R_COL[0] + bar_w - PCT_BUFFER
                ha = 'right'
            else:
                x = R_COL[0] + bar_w + PCT_BUFFER
                ha = 'left'
    
            ax_rep.text(
                x,
                y,
                f"{row['R']:.0f}%",
                ha=ha,
                va='center',
                fontsize=FONT_SIZE,
            )
            
            draw_bar(
                ax_rep,
                CR_COL[0],
                R_COL[0],
                y,
                1,
                color='#FFFFFF',
                height=row_h,
                rounded=False,
                alpha=1
            )
    
            ax_rep.text(
                sum(CR_COL) / 2,
                y,
                f"{row['Shape+ R']:.0f}",
                ha='center',
                va='center',
                weight='bold',
                fontsize=FONT_SIZE,
            )


    
    #
    # Credits
    #
    
    ax_cred.axis('off')
    ax_cred.text(0.5, 0.3, 'Data and visualization are property of Major League Baseball and the Chicago White Sox.', ha='center', va='center', fontsize=10)
    
    # final display
    plt.show()
    return
profile_viz(query_grades, query_repertoire, classified_query_data, conditions, no_slot)

def visualize_slot_effects(pbp_data, pitch_type, dpi=150):    

    ipython = get_ipython()
    ipython.run_line_magic('matplotlib', 'qt')
    
    if len(pbp_data) == 0:
        print('\nNo slot data found.')
        return
    
    eff_dict = joblib.load('Data/eff_dict')
    
    player_name = pbp_data['player_name']['first'].values[0]
    name = player_name.split(', ')[1] + ' ' + player_name.split(', ')[0]
    
    repertoire = pbp_data['true_pitch_type'].unique()
    
    if pitch_type in repertoire:
        df = pbp_data[pbp_data['true_pitch_type'] == pitch_type].copy()
    else:
        print(f'\nNo {pitch_type.lower()}s found for {name}. His repertoire is:\n\n{', '.join(repertoire)}')
        return
    
    hand = df['p_throws']['first'].values[0]


    
    def plot_movement_effectiveness_combined(data, effectiveness_map, cluster, pitch_type, pitcher=None):
        
        df = effectiveness_map.copy()
        df['mean'] = 3000 * df['mean']
        
        n_samples = df['count'].sum()
        sample_size = round(math.log10(n_samples))
        smoothing_sigma = 7.5/sample_size
        
        # First, create a regular grid
        x_edges = np.unique(np.concatenate([df['pfx_x_min'], df['pfx_x_max']]))
        z_edges = np.unique(np.concatenate([df['pfx_z_min'], df['pfx_z_max']]))
                
        # Get the bin centers for plotting
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        
        # Create empty grids for values, counts, and masks
        grid_values = np.full((len(z_centers), len(x_centers)), np.nan)
        grid_counts = np.zeros_like(grid_values)
        mask = np.full_like(grid_values, False, dtype=bool)
        
        # Fill the grids with values and counts
        for _, row in df.iterrows():
            x_idx = np.where(np.isclose(x_centers, (row['pfx_x_min'] + row['pfx_x_max']) / 2))[0][0]
            z_idx = np.where(np.isclose(z_centers, (row['pfx_z_min'] + row['pfx_z_max']) / 2))[0][0]
            grid_values[z_idx, x_idx] = row['mean']
            grid_counts[z_idx, x_idx] = row['count']
            mask[z_idx, x_idx] = True
        
        # Create weighted data for smoothing
        weighted_data = np.copy(grid_values)
        weighted_data[~mask] = 0
        weighted_data = weighted_data * grid_counts
        
        # Smooth both the weighted data and the counts
        smooth_weighted_data = gaussian_filter(weighted_data, sigma=smoothing_sigma)
        smooth_counts = gaussian_filter(grid_counts, sigma=smoothing_sigma)
        
        # Calculate the smoothed values, normalized by counts
        with np.errstate(divide='ignore', invalid='ignore'):
            grid_smooth = np.where(smooth_counts > 5,  # Threshold to avoid division by small numbers
                                 smooth_weighted_data / smooth_counts,
                                 np.nan)
        
        # Center the values to ensure weighted average is zero
        valid_mask = ~np.isnan(grid_smooth)
        if np.any(valid_mask):
            # Calculate current weighted average
            current_weighted_avg = np.average(
                grid_smooth[valid_mask],
                weights=smooth_counts[valid_mask]
            )
            # Subtract the weighted average to center at zero
            grid_smooth[valid_mask] -= current_weighted_avg
            
            # Verify the new weighted average is approximately zero
            new_weighted_avg = np.average(
                grid_smooth[valid_mask],
                weights=smooth_counts[valid_mask]
            )
            assert abs(new_weighted_avg) < 1e-5, "Weighted average is not zero after centering"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(7, 5.33), dpi=dpi, layout="constrained")
        
        # Set fixed colormap limits
        vmin, vmax = -10, 10  # Fixed run value limits
        norm = plt.Normalize(vmin, vmax)
        
        # Create the heatmap using pcolormesh
        mesh = ax.pcolormesh(
            x_edges, 
            z_edges, 
            grid_smooth,
            cmap='bwr',
            norm=norm,
            shading='flat'
        )
        
        # Add labels and title
        min_samples = df['count'].min()
        value = round((np.max(grid_smooth) - np.min(grid_smooth))/2)

        velo = cluster[0]
        pitch_class = cluster[1]
        display_dict = {'fastball': 'fastballs',
                        'sinker':   'sinkers',
                        'breaking': 'breaking balls',
                        'offspeed': 'offspeed pitches'}
        if pitcher is not None:   
            hand = data['p_throws']['first'].values[0]
            slot = round(data['arm_angle']['mean'].values[0])
            pfirst = pitcher.split(', ')[1]
            plast = pitcher.split(', ')[0]
            fig.suptitle(f"Expected Run Value by Arm Slot Deviation for\n{pfirst} {plast}'s {pitch_type} ({hand}HP, {slot}" + r"$\degree$)", fontsize=16)
            ax.set_title(f'Graded against other {velo-1}-{velo+1} mph {display_dict[pitch_class]}', fontsize = 14, style='italic')
            ax.set_xlabel('Unexpected Horizontal Break (Pitcher POV)')
            ax.set_ylabel('Unexpected Vertical Break (in.)')
            
        else:
            ax.set_title(f'{pitch_type} xRV by Arm Slot Deviation (Min. {min_samples} samples per bin)\n\u00B1{value} runs per 3000 pitches available due to slot deviation.')
            ax.set_xlabel('Unexpected Horizontal Break (inches from slot-derived median, pitcher POV)')
            ax.set_ylabel('Unexpected Vertical Break (inches from median)')
        
        w = 7 if pitch_class in ('fastball', 'sinker') else 11
        ax.set_xlim((-w, w))
        ax.set_ylim((-w, w))
            
        # Add reference lines
        lw = 1/72
        ax.axhline(y=-lw/2, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=lw/2, color='k', linestyle='--', alpha=0.3)
        
        # Add colorbar with fixed limits
        if pitcher:
            pass
        else:
            fig.colorbar(mesh, ax=ax, label='xRV per 3000 pitches', shrink= 0.7, ticks = [-5, 0, 5])
        
        ax.set_aspect(1)
        
        if pitcher:
            return fig, ax 
        else:
            plt.show()
            
        return
    
    n_pitches = 1000

    xRV = round(df['diff_rv']['mean'].values[0]*n_pitches, 1)
    
    if xRV > 0:
        xRV = '+' + str(round(xRV, 1))
    else:
        xRV = str(round(xRV, 1))
    
    x0 = df['x_zero']['mean'].values[0]
    z0 = df['z_zero']['mean'].values[0]

    def x_abs(x_rel):
        return x_rel + x0
    def x_rel(x_abs):
        return x_abs + x0
    def z_abs(z_rel):
        return z_rel + z0
    def z_rel(z_abs):
        return z_abs + z0


    pitch_class = df['pitch_class'].values[0]
    velo = df['release_speed']['mean'].values[0]
    velo = int(3 * round(velo/3))
    cluster = [velo, pitch_class]
    
    # ensure valid cluster
    try:
        effectiveness_map = eff_dict[velo][pitch_class]
    except KeyError:
        display_dict = {'fastball': 'fastballs',
                        'sinker':   'sinkers',
                        'breaking': 'breaking balls',
                        'offspeed': 'offspeed pitches'}
        print(f"\nThis pitch is too unique; there have not been enough {display_dict[pitch_class]}\nthrown at {velo} mph. Slot effects cannot be accurately measured.")
        return
        
    fig, ax = plot_movement_effectiveness_combined(df, effectiveness_map, cluster, pitch_type, pitcher=player_name)
    
    xax = ax.secondary_xaxis('top', functions=(x_abs, x_rel))
    xax.set_xlabel('Actual Horizontal Break (in.)')
    zax = ax.secondary_yaxis('right', functions=(z_abs, z_rel))
    zax.set_ylabel('Actual Inducued Vertical Break (in.)')
    
    x_mean = df['x_rel']['mean'].values[0]
    x_std = df['x_rel']['std'].values[0]
    z_mean = df['z_rel']['mean'].values[0]
    z_std = df['z_rel']['std'].values[0]
    
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = x_mean + x_std * np.cos(theta)
    ellipse_z = z_mean + z_std * np.sin(theta)
    
    # Plot the ellipse
    ax.fill(ellipse_x, ellipse_z, alpha=0.5, fc='k')
    ax.scatter(x_mean, z_mean, s = 50, color = 'k', label = xRV + ' xRV\nper 1000 pitches')
    ax.legend(fontsize=14)
        
    # ensure plot is large enough
    xlims = ax.get_xlim()
    zlims = ax.get_ylim()
    
    if x_mean < xlims[0]:
        expansion_factor = (xlims[1] - x_mean) / (xlims[1] - xlims[0])
        ax.set_xlim((x_mean, xlims[1]))
        ax.set_ylim((zlims[0] * expansion_factor, zlims[1] * expansion_factor))
    if x_mean > xlims[1]:
        expansion_factor = (x_mean - xlims[0]) / (xlims[1] - xlims[0])
        ax.set_xlim((xlims[0], x_mean))
        ax.set_ylim((zlims[0] * expansion_factor, zlims[1] * expansion_factor/2))
    if z_mean < zlims[0]:
        expansion_factor = (zlims[1] - z_mean) / (zlims[1] - zlims[0])
        ax.set_ylim((z_mean, zlims[1]))
        ax.set_xlim((xlims[0] * expansion_factor, xlims[1] * expansion_factor))
    if z_mean > zlims[1]:
        expansion_factor = (z_mean - zlims[0]) / (zlims[1] - zlims[0])
        ax.set_ylim((zlims[0], z_mean))
        ax.set_xlim((xlims[0] * expansion_factor, xlims[1] * expansion_factor))

    if hand == 'L':
        plt.gca().invert_xaxis()
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    xax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    zax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
    
    ax.set_aspect(1)
    plt.show()        
    
    return
# visualize_slot_effects(query_slot_info, 'Riding Fastball')
# visualize_slot_effects(query_slot_info, 'Fastball')
# visualize_slot_effects(query_slot_info, 'Sinker')
# visualize_slot_effects(query_slot_info, 'Cutter')
# visualize_slot_effects(query_slot_info, 'Gyro Slider')
# visualize_slot_effects(query_slot_info, 'Carry Slider')
# visualize_slot_effects(query_slot_info, 'Two-Plane Slider')
# visualize_slot_effects(query_slot_info, 'Sweeper')
# visualize_slot_effects(query_slot_info, 'Slurve')
# visualize_slot_effects(query_slot_info, 'Curveball')
# visualize_slot_effects(query_slot_info, 'Movement-Based Changeup')
# visualize_slot_effects(query_slot_info, 'Velo-Based Changeup')



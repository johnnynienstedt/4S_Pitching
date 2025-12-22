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



import joblib
import pandas_gbq
import pydata_google_auth
import numpy as np
import pandas as pd
from numba import jit
from unidecode import unidecode
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

#%%
def get_data(sf_level='mlb', league=None, team=None, org=None, start_dt=None, end_dt=None, game_type=None):
    
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
    
    # optional league condition
    if league is not None:
        if type(league) == str:
            league_condition = f"sc.teams.home.team.league.name = '{league}'"
        elif type(league) == tuple:
            league_condition = f"sc.teams.home.team.league.name in {league}"
        else:
            raise ValueError("'league' must be a string or tuple of strings.")
    else:
        league_condition = "1 = 1"
    
    # optional team(s) condtion
    if team is not None:  
        if type(team) == str:
            team_condition = f"pp.current_team = '{team}'"
        elif type(team) == tuple:
            team_condition = f"pp.current_team in {team}"
        else:
            raise ValueError("'team' must be a string or tuple of strings.")
    else:
        team_condition = 'pp.current_team is not NULL'
    
    # optional org condition
    if org is not None:
        if type(org) == str:
            org_condition = f"pp.current_org = '{org}'"
        else:
            raise ValueError("Please input a single organization. This script does not support querying multiple organizations at once.")
    else:
        org_condition = "pp.current_org is not NULL"
    
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
    
    conditions = {'sf_level': sf_level,
                  'league': league,
                  'team': team,
                  'start_dt': start_dt, 
                  'end_dt': end_dt, 
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
    LEFT JOIN `sfgiants-viewer.scout_replica_scouting.tblPlayer_Pro` as pp
    ON ge.pitcher_sfg_id = pp.player_code
    LEFT JOIN `prod-warehouse-external.stats_api_v2.schedule` as sc
    ON ge.game_pk = sc.gamePk
    
    WHERE 
    {level_condition} and
    {league_condition} and
    {team_condition} and
    {org_condition} and
    {date_condition} and
    {type_condition}
    """
    
    all_pitch_data = pandas_gbq.read_gbq(four_s_query,
                                         project_id="sfgiants-viewer",
                                         credentials=credentials)
    
    if len(all_pitch_data) == 0:
        raise ValueError("No data for given query.")
        
    # fill missing values
    if sf_level in ('intl', 'am'):
        all_pitch_data['pitcher'] = all_pitch_data['player_name']
        all_pitch_data['game_pk'] = all_pitch_data['game_pk'].fillna(1)
        all_pitch_data['game_type'] = all_pitch_data['game_type'].fillna('R')
        all_pitch_data['is_ball_in_play'] = all_pitch_data['is_ball_in_play'].fillna(False)
        
    if game_type == 'all':
        all_pitch_data['pitcher'] = all_pitch_data['player_name']
        all_pitch_data['game_pk'] = all_pitch_data['game_pk'].fillna(1)
        all_pitch_data['game_type'] = all_pitch_data['game_type'].fillna('X')

    # get macro results
    results_list = []
    for pitcher, pitcher_data in all_pitch_data.groupby('pitcher'):
        bf_per_g = pitcher_data.groupby(['game_year', 'month', 'day'])['is_last_pitch_plate_appearance'].sum().mean()
        starter = 1 if bf_per_g > 12 else 0
        
        tbf = sum(pitcher_data['is_last_pitch_plate_appearance'].dropna())
        k_pct = sum(pitcher_data['is_strikeout'].dropna()) / tbf
        bb_pct = sum(pitcher_data['is_base_on_balls'].dropna()) / tbf
        gbs = sum((pitcher_data['is_ball_in_play']) & (pitcher_data['hit_exit_angle'] < 10))
        fbs = sum((pitcher_data['is_ball_in_play']) & (pitcher_data['hit_exit_angle'].between(25, 50)))
        pus = sum((pitcher_data['is_ball_in_play']) & (pitcher_data['hit_exit_angle'] > 50))
        
        bip = (gbs - fbs - pus) / tbf
        pm = -1 if bip > 0 else 1
        siera = 6.645 - 16.986 * k_pct + 11.434 * bb_pct - 1.858 * bip + 7.653 * k_pct**2 + pm * 6.664 * bip**2 + 10.130 * k_pct * bip - 5.195 * bb_pct * bip
        
        results_dict = {
            'pitcher': pitcher,
            'starter': starter,
            'K%': k_pct,
            'BB%': bb_pct,
            'SIERA': siera
        }
        results_list.append(results_dict)
    results_data = pd.DataFrame(results_list)
    
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
    
    # select pitchers with at least 100 pitches thrown
    pitch_data = clean_data.groupby(['pitcher']).filter(lambda x: len(x) >= 100)
    
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
classified_query_data, results_data, conditions = get_data(sf_level=None,
                                                           org='San Diego Padres',
                                                           start_dt='2025-01-01',
                                                           end_dt='2025-12-31',
                                                           game_type=None)
#%%
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

    # grouped = grading_data.groupby(['pitcher', 'true_pitch_type', 'platoon'])[display_cols]
    # repertoire = grouped.mean().reset_index().copy()
    
    # counts = grouped.size().rename("count")
    # repertoire = repertoire.merge(counts, on=['pitcher', 'true_pitch_type', 'platoon'])
    
    # platoon_rate = grading_data['platoon'].mean()
    
    # percentages = grouped.size() / len(grading_data)
    
    # repertoire['percent'] = (
    #     repertoire.set_index(['pitcher', 'true_pitch_type', 'platoon']).index.map(percentages)
    # )
    
    # repertoire['percent'] = (
    #     repertoire['percent'] * 100 /
    #     repertoire['platoon'].map({0: 1 - platoon_rate, 1: platoon_rate})
    # )
    
    # # for convenience, ensure count column stays early in column order
    # cols = list(repertoire.columns)
    # cols.insert(4, cols.pop(cols.index('count')))
    # repertoire = repertoire[cols]

    # # rename
    # repertoire.rename(columns={col: col.replace('predicted_', '') for col in repertoire.columns}, inplace=True)



    '''
    ###############################################################################
    ################################ By Pitch Type ################################
    ###############################################################################
    '''

    # grouped = grading_data.groupby(['pitcher', 'true_pitch_type'])[display_cols]
    # pt_shape_grades = grouped.mean().reset_index().copy()
    
    # counts = grouped.size().rename("count")
    # pt_shape_grades = pt_shape_grades.merge(counts, on=['pitcher', 'true_pitch_type'])
    
    # # for convenience, ensure count column stays early in column order
    # cols = list(pt_shape_grades.columns)
    # cols.insert(2, cols.pop(cols.index('count')))
    # pt_shape_grades = pt_shape_grades[cols]



    '''
    ###############################################################################
    ############################### By Pitcher Only ###############################
    ###############################################################################
    '''
    
    grouped = grading_data.groupby(['pitcher', 'player_name'])[display_cols]
    shape_grades = grouped.mean().reset_index().copy()
    shape_grades.insert(1, 'count', grouped.size().values)
    shape_grades.insert(2, 'true_pitch_type', 'total')    

    # # concatenate
    # shape_grades = pd.concat([shape_grades, pt_shape_grades], ignore_index=True)

    # rename
    shape_grades.rename(columns={col: col.replace('predicted_', '') for col in shape_grades.columns}, inplace=True)
    
    # for aesthetic purposes
    def rearrange_name(name):
        last, first = name.split(', ')[:2]
        return f"{first} {last}"
    
    shape_grades.insert(0, 'Name', shape_grades['player_name'].apply(rearrange_name))
    
    return shape_grades
query_shape_grades = grade_shape(classified_query_data)
#%%
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
        pitcher_groups = df.groupby('pitcher')
        
        slot_variance_df = pd.DataFrame(
            index=df['pitcher'].unique(),
            columns=['pitcher', 'Name', 'release_variance', 'repertoire_variance', 'pbp_variance'])
        slot_variance_df['pitcher'] = df['pitcher'].unique()
        slot_variance_df.sort_index(inplace=True)
    
        for pitcher, row in slot_variance_df.iterrows():
            
            try:
                df = pitcher_groups.get_group(pitcher)
            except KeyError:
                continue

            if len(df) < min_pitches:
                continue
        
            # Set name
            last, first = df['player_name'].iloc[0].split(', ')[:2]
            slot_variance_df.loc[pitcher, 'Name'] = f"{first} {last}"
            
            # Calculate overall release variance using combined std of horizontal and vertical angles
            release_points = df[['release_angle_h', 'release_angle_v']].to_numpy()
            total_std = np.sum(np.std(release_points, axis=0))
            slot_variance_df.loc[pitcher, 'release_variance'] = total_std
            
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
                slot_variance_df.loc[pitcher, 'repertoire_variance'] = repertoire_variance
                
                # Calculate pbp variance as weighted average of per-pitch type stds
                total_pitches = sum(stat['count'] for stat in pitch_type_stats)
                weighted_std = sum(stat['std'] * stat['count'] for stat in pitch_type_stats) / total_pitches
                slot_variance_df.loc[pitcher, 'pbp_variance'] = weighted_std
    
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
    
    # #
    # # Pitch type level
    # #
    
    # pitchtype_df = (
    #     slot_data
    #     .groupby(['pitcher', 'true_pitch_type'])
    #     .agg(agg_dict)
    #     .reset_index()
    # )
    # pitchtype_df['count'] = slot_data.groupby(['pitcher', 'true_pitch_type']).size().values
    
    # loc_grades = pd.concat([pitcher_df, pitchtype_df], ignore_index=True)
    loc_grades = pitcher_df.copy()
    
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
#%%
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
    # pt_slot_grades = grade_data.groupby(['pitcher', 'true_pitch_type']).agg(agg_dict).dropna().reset_index()
    
    # slot_grades = pd.concat([slot_grades, pt_slot_grades], ignore_index=True)
    
    # # for visualization
    # pbp_cols = ['release_speed', 'x_zero', 'z_zero', 'x_rel', 'z_rel']
    # agg_dict.update({col: ['mean', 'std'] for col in pbp_cols})
    # agg_dict.update({'slot_cluster': 'count', 'p_throws': 'first'})
    # slot_info = grade_data.groupby(['pitcher', 'true_pitch_type', 'pitch_class'], observed=True).agg(agg_dict).dropna().reset_index()
    # slot_info = slot_info.rename(columns = {'slot_cluster': 'count'})
    
    return slot_grades
query_slot_grades = grade_slot(classified_query_data)
#%%
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
    # sequence_grades_1_pt = classified_pitch_data.groupby(['pitcher', 'true_pitch_type']).agg(agg_dict).reset_index()
    
    # sequence_grades_1 = pd.concat([sequence_grades_1, sequence_grades_1_pt])
    
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
    # sequence_grades_2_pt = classified_pitch_data.groupby(['pitcher', 'true_pitch_type']).agg(agg_dict).reset_index()

    # sequence_grades_2 = pd.concat([sequence_grades_2, sequence_grades_2_pt])
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
#%%
def predict_performance(shape_grades, spot_grades, slot_grades, sequence_grades, results_data):
    
    KEYS = ['pitcher']
    
    if len(slot_grades) != len(spot_grades):
        if len(slot_grades) == 0:
            slot_grades = spot_grades.copy()
            slot_grades['diff_rv'] = 0
            slot_grades['slot_rarity'] = 0.114
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

    # # Shape+ only by handedness
    # def get_shape(repertoire):
        
    #     df = repertoire.copy()
        
    #     xcols = ['swstr', 'foul', 'weak', 'topped', 'under', 'flr_brn', 'solid', 'barrel']
            
    #     x_norms = joblib.load('Data/x_norms_pred')
    #     x_means = x_norms['mean']
    #     x_stds = x_norms['std']
    #     df[xcols] = normalize(df[xcols], 0, 1, pop_mean=x_means, pop_std=x_stds, dtype=float)
        
    #     results = joblib.load('Data/SIERA_prediction_weights')
        
    #     shape_norms = joblib.load('Data/shape_norms')
    #     shape_mean = shape_norms['mean']
    #     shape_std = shape_norms['std']
            
    #     repertoire['Shape+'] = normalize(-sum(df[col] * results['coefficients'].get(col, 0) for col in xcols), 100, 10, pop_mean=shape_mean, pop_std=shape_std, dtype=float)
    #     repertoire['Shape+'] = np.round((100 * 10 + repertoire['Shape+'] * repertoire['count']) / (10 + repertoire['count'])).astype(int)

    #     return repertoire
    
    # graded_repertoire = get_shape(repertoire)

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

    return grades
query_grades = predict_performance(query_shape_grades, query_spot_grades, query_slot_grades, query_sequence_grades, results_data)


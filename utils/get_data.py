import pandas as pd
import requests
from feature_selection_config import fpl_features_by_position, fpl_lagged_features_by_position, fpl_features_to_smooth_by_position


def get_upcoming_fixture_data(num_fixtures: int, bootstrap_data: dict) -> dict:

    df_teams = pd.DataFrame(bootstrap_data['teams'])
    current_gw = next(event['id'] for event in bootstrap_data['events'] if event['is_current'])

    gw_fixtures_list = []

    #getting the fixtures for the upcoming gameweeks
    for gw in range(current_gw, current_gw + num_fixtures):
        fixtures_url = f"https://fantasy.premierleague.com/api/fixtures/?event={gw}"
        gw_response = requests.get(fixtures_url)
        if gw_response.status_code != 200:
            raise Exception(f"Failed to fetch data from {fixtures_url}")
        gw_fixtures_list.append(gw_response.json())
    
    #team_info_dictionary = {team_id : pd.DataFrame[was_home, strength_difference, attack_strenght_difference, defence_strenght_difference], [...] ...}
    team_info_dictionary = {}

    for team_id in df_teams['id']:
        team_info_dictionary[team_id] = pd.DataFrame(columns=[
            'was_home',
            'strength_difference',
            'attack_strength_difference',
            'defense_strength_difference'
        ])

    for fixtures in gw_fixtures_list: 
        for fixture in fixtures:
            team_info_dictionary[fixture['team_h']].append([
                True,
                df_teams.loc[df_teams['id'] == fixture['team_h'], 'strength'].values[0] - df_teams.loc[df_teams['id'] == fixture['team_a'], 'strength'].values[0],
                df_teams.loc[df_teams['id'] == fixture['team_h'], 'strength_attack_home'].values[0] - df_teams.loc[df_teams['id'] == fixture['team_a'], 'strength_defence_away'].values[0],
                df_teams.loc[df_teams['id'] == fixture['team_h'], 'strength_defence_home'].values[0] - df_teams.loc[df_teams['id'] == fixture['team_a'], 'strength_attack_away'].values[0]
            ])
            team_info_dictionary[fixture['team_a']].append([
                False,
                df_teams.loc[df_teams['id'] == fixture['team_a'], 'strength'].values[0] - df_teams.loc[df_teams['id'] == fixture['team_h'], 'strength'].values[0],
                df_teams.loc[df_teams['id'] == fixture['team_a'], 'strength_attack_away'].values[0] - df_teams.loc[df_teams['id'] == fixture['team_h'], 'strength_defence_home'].values[0],
                df_teams.loc[df_teams['id'] == fixture['team_a'], 'strength_defence_away'].values[0] - df_teams.loc[df_teams['id'] == fixture['team_h'], 'strength_attack_home'].values[0]
            ])
    
    return team_info_dictionary






def get_historical_player_data(position) -> pd.DataFrame:

    # Return a DataFrame with historical player data for the given position
    
    # Getting the relevant features for the given position from a dictionary in the feature_selection_config.py file
    features = fpl_features_by_position[position],

    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv"

    #getting data from the 25/26 season 
    try:
        df = pd.read_csv(url, usecols=feature)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data for the {position} position not found in the specified URL: {url}")
    
    df = df[df['position'] == position]

    num_gws = df['GW'].nunique()   
    #if the number of gameweeks is less than 5, we need to get the data from the 24/25 season as well 
    NUM_LAGS = 5 
    if num_gws < NUM_LAGS:
        df['season'] = '2024-25'
        url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv"
        
        df_24_25 = pd.read_csv(url, usecols=features)
        df_24_25 = df_24_25[df_24_25['name'].isin(df['name'].unique())]
        df_24_25.drop(df_24_25[df_24_25['GW'] <= 38 - (NUM_LAGS - num_gws)].index, inplace=True)        
        df_24_25['season'] = '2024-25'
        
        df = pd.concat([df, df_24_25])
        df.sort_values(by=['name', 'season', 'GW'], ascending=[True, True, True], inplace=True)
        df.drop('season', axis=1, inplace=True)

    else:
        df.drop(df[df['GW'] <= num_gws - NUM_LAGS].index, inplace=True)
        df.sort_values(by=['name', 'GW'], ascending=[True, True], inplace=True)

    df.drop_duplicates(subset=['name', 'GW'], keep='first', inplace=True)

    # Engineer Features for Predicting the VARIANCE (scale)
    # Rolling Standard Deviation (Volatility)
    df['points_std_roll5'] = df.groupby('name')['total_points'].transform(
        lambda x: x.rolling(window=NUM_LAGS, min_periods=1).std()
    )
    # Haul & Blank Counts (Boom-or-Bust Metric)
    HAUL_THRESHOLD = 8
    BLANK_THRESHOLD = 3
    df['hauls_roll5'] = df.groupby('name')['total_points'].transform(
        lambda x: x.rolling(window=NUM_LAGS, min_periods=1).apply(lambda y: (y >= HAUL_THRESHOLD).sum())
    )

    df['blanks_roll5'] = df.groupby('name')['total_points'].transform(
        lambda x: x.rolling(window=NUM_LAGS, min_periods=1).apply(lambda y: (y < BLANK_THRESHOLD).sum())
    )

    features_to_smooth = fpl_features_to_smooth_by_position[position]
    for feature in features_to_smooth:
        df[f'{feature}_ewma'] = df.groupby('name')[feature].transform(
            lambda x: x.ewm(span=NUM_LAGS, adjust=False).mean()
        )

    features_to_lag = fpl_lagged_features_by_position[position]

    NUM_LAGS = 2
    for feature in features_to_lag:
        for i in range(1, NUM_LAGS + 1):
            # Create a new column with lagged values
            df[f'{feature}_lag{i}'] = df.groupby('name')[feature].shift(i-1)
    
    # Finalize Feature Set 
    ewma_cols = [f'{f}_ewma' for f in features_to_smooth]
    lag_cols = []
    for i in range(1, NUM_LAGS + 1):
        lag_cols.extend([f'{f}_lag{i}' for f in features_to_lag])

    variance_cols = ['points_std_roll5', 'hauls_roll5', 'blanks_roll5']
    metadata_cols = ['name']

    features_to_keep = ewma_cols + lag_cols + variance_cols + metadata_cols
    df_final = df[features_to_keep].copy()
    df_final = df_final.fillna(0) 
    
    return df_final

def get_player_metadata(bootstrap_data: dict) -> pd.DataFrame:

    metadata_cols = ['first_name', 'second_name', 'id', 'team', 'element_type', 'now_cost', 'chance_of_playing_this_round', 'chance_of_playing_next_round']
    
    df = pd.DataFrame(bootstrap_data['elements'], columns=metadata_cols)
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)

    df['name'] = df['first_name'] + ' ' + df['second_name']
    df.drop(['first_name', 'second_name'], axis=1, inplace=True)

    element_type_to_position = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    df['position'] = df['element_type'].map(element_type_to_position)
    df.drop('element_type', axis=1, inplace=True)
    
    return df
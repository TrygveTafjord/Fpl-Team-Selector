import pandas as pd
from urllib.error import URLError
import requests
from feature_selection_config import fpl_features_by_position, fpl_lagged_features_by_position, fpl_features_to_smooth_by_position
import collections


def get_upcoming_fixture_data(num_fixtures: int, bootstrap_data: dict) -> dict:

    all_fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(all_fixtures_url)
    response.raise_for_status() # Replaces the need for status code check
    all_fixtures = response.json()

    df_teams = pd.DataFrame(bootstrap_data['teams']).set_index('id')
    current_gw = next(event['id'] for event in bootstrap_data['events'] if event['is_current'])

    # 2. Filter fixtures for the desired gameweeks
    upcoming_gws = range(current_gw, current_gw + num_fixtures)
    upcoming_fixtures = [f for f in all_fixtures if f['event'] in upcoming_gws]

    team_info_dictionary = collections.defaultdict(list)

    for fixture in upcoming_fixtures:
        # Get team IDs
        home_team_id = fixture['team_h']
        away_team_id = fixture['team_a']

        # Using index for fast, direct lookups
        home_stats = df_teams.loc[home_team_id]
        away_stats = df_teams.loc[away_team_id]

        # append data for the home
        team_info_dictionary[home_team_id].append([
            True,
            home_stats['strength'] - away_stats['strength'],
            home_stats['strength_attack_home'] - away_stats['strength_defence_away'],
            home_stats['strength_defence_home'] - away_stats['strength_attack_away']
        ])

        # append data for the away team 
        team_info_dictionary[away_team_id].append([
            False,
            away_stats['strength'] - home_stats['strength'],
            away_stats['strength_attack_away'] - home_stats['strength_defence_home'],
            away_stats['strength_defence_away'] - home_stats['strength_attack_home']
        ])
        
    return team_info_dictionary



def get_historical_player_data(position) -> pd.DataFrame:
        
    # Getting the relevant features for the given position from a dictionary in the feature_selection_config.py file
    features = fpl_features_by_position[position]

    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv"

    #getting data from the 25/26 season 
    try:
        df = pd.read_csv(url, usecols=features)
    except URLError:
        raise URLError(f"Data for the {position} position not found in the specified URL: {url}")
    
    df = df[df['position'] == position]

    num_gws = df['GW'].nunique()   
    #if the number of gameweeks is less than 5, we need to get the data from the 24/25 season as well 
    
    ROLLING_WINDOW = 5
    FEATURE_LAG_COUNT = 2
    NUM_LAGS = max(ROLLING_WINDOW, FEATURE_LAG_COUNT)
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
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).apply(lambda y: (y >= HAUL_THRESHOLD).sum())
    )

    df['blanks_roll5'] = df.groupby('name')['total_points'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).apply(lambda y: (y < BLANK_THRESHOLD).sum())
    )

    features_to_smooth = fpl_features_to_smooth_by_position[position]
    for feature in features_to_smooth:
        df[f'{feature}_ewma'] = df.groupby('name')[feature].transform(
            lambda x: x.ewm(span=ROLLING_WINDOW, adjust=False).mean()
        )

    features_to_lag = fpl_lagged_features_by_position[position]

    for feature in features_to_lag:
        for i in range(1, FEATURE_LAG_COUNT + 1):
            # Create a new column with lagged values
            df[f'{feature}_lag{i}'] = df.groupby('name')[feature].shift(i-1)
    
    # Finalize Feature Set 
    ewma_cols = [f'{f}_ewma' for f in features_to_smooth]
    lag_cols = []
    for i in range(1, FEATURE_LAG_COUNT + 1):
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
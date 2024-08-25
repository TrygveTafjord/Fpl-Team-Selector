import pandas as pd
import requests
from feature_selection_config import fpl_features_by_position, fpl_lagged_features_by_position

# Constants for the team_info_dictionary
WAS_HOME = 0
STRENGHT_DIFFERENCE = 1 
ATTACK_STRENGHT_DIFFERENCE = 2
DEFENSE_STRENGHT_DIFFERENCE = 3

def get_upcoming_fixture_data(num_fixtures) -> dict:
    """
    "input": num_fixtures - int - the number of upcoming fixtures we want to get the data from
    "output": info of upcoming fixtures - dict - a dictionary where each key is a club id, and the value is a list of dictionaries containing match data for the upcoming num_fixtures 
    """
    #getting team info and current gameweek 
    fpl_api_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    static_response = requests.get(fpl_api_url)
    if static_response.status_code != 200:
        raise Exception(f"Failed to fetch data from {fpl_api_url}")

    df_teams = pd.DataFrame(static_response.json()['teams'])

    current_gw = next(event['id'] for event in static_response.json()['events'] if event['is_current'])

    gw_fixtures_list = []

    #getting the fixtures for the upcoming gameweeks
    for gw in range(current_gw, current_gw + num_fixtures):
        fixtures_url = f"https://fantasy.premierleague.com/api/fixtures/?event={gw}"
        gw_response = requests.get(fixtures_url)
        if gw_response.status_code != 200:
            raise Exception(f"Failed to fetch data from {fixtures_url}")
        gw_fixtures_list.append(gw_response.json())
    
    #creating a dictionary with the team id as key and the strength of the team as value
    #team_info_dictionary = {team_id : [was_home, strength_difference, attack_strenght_difference, defence_strenght_difference], [...] ...}
    team_info_dictionary = {}

    for team_id in df_teams['id']:
        team_info_dictionary[team_id] = []

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



def get_last_five_matches_player_data(position) -> pd.DataFrame:
    """
    "input": position - str - the position of the players we want to get the data from
    "output": pd.DataFrame - the data of the last five matches of the players of the given position as lagged features
    """
    NUM_LAGS = 5 
    # Getting the relevant features for the given position from a dictionary in the feature_selection_config.py file
    relevant_features = fpl_features_by_position[position]

    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv"

    #getting data from the 24/25 season 
    df = pd.read_csv(url, usecols=relevant_features)
    df = df[df['position'] == position]

    num_gws = df['GW'].nunique()   
    #if the number of gameweeks is less than 5, we need to get the data from the 23/24 season as well 
    if num_gws < NUM_LAGS:
        df['season'] = '2024-25'
        url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv"
        
        df_23_24 = pd.read_csv(url, usecols=relevant_features)
        df_23_24 = df_23_24[df_23_24['name'].isin(df['name'].unique())]
        df_23_24.drop(df_23_24[df_23_24['GW'] <= 38 - (NUM_LAGS - num_gws)].index, inplace=True)        
        df_23_24['season'] = '2023-24'
        
        df = pd.concat([df, df_23_24])
        df.sort_values(by=['name', 'season', 'GW'], ascending=[True, True, True], inplace=True)
        df.drop('season', axis=1, inplace=True)

    else:
        df.drop(df[df['GW'] <= num_gws - NUM_LAGS].index, inplace=True)
        df.sort_values(by=['name', 'GW'], ascending=[True, True], inplace=True)

    df.drop_duplicates(subset=['name', 'GW'], keep='first', inplace=True)

    # Creating lagged features
    lagged_features = fpl_lagged_features_by_position[position]
    
    combined_rows = []

    for name, group in df.groupby('name'):
            combined_row = {}
            combined_row['name'] = name
            for i in range(NUM_LAGS):
                if i < len(group):
                    row = group.iloc[i]
                    for feature in lagged_features:
                        combined_row[f'{feature}_lag{i+1}'] = row[feature]
                else:
                    for feature in lagged_features:
                        combined_row[f'{feature}_lag{i+1}'] = pd.NA

            combined_rows.append(combined_row)

    df_combined = pd.DataFrame(combined_rows)

    return df_combined

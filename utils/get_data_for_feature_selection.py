import pandas as pd

def get_player_data_by_pos(position: str, relevant_features: list[str]) -> pd.DataFrame:
    
    # Getting data from 22/23 season    
    df_22_23 = pd.read_csv("../data/2022-23/gws/merged_gw.csv", usecols=relevant_features)
    df_22_23['season'] = '2022-23'

    # Getting data from 23/24 season
    df_23_24 = pd.read_csv("../data/2023-24/gws/merged_gw.csv", usecols=relevant_features)
    df_23_24['season'] = '2023-24'
    
    # Merging the data
    df = pd.concat([df_22_23, df_23_24])

    # Filtering only the given position
    df = df[df['position'] == position]

    # Adding information about the opponent team, looking at strength differences
    team_info_cols = [
                     'id', 'name', 'strength', 'strength_attack_home', 'strength_attack_away', 
                     'strength_defence_home', 'strength_defence_away'
                     ]
    
    team_info_df_22_23 = pd.read_csv(f"../data/2022-23/teams.csv", usecols=team_info_cols)
    team_info_df_23_24 = pd.read_csv(f"../data/2023-24/teams.csv", usecols=team_info_cols)

    for index, row in df.iterrows():

        if row['season'] == '2022-23':
            team_info_df = team_info_df_22_23
        else:
            team_info_df = team_info_df_23_24

        if row['was_home']:
            df.at[index, 'attack_strenght_difference'] = team_info_df.loc[team_info_df['name'] == row['team'], 'strength_attack_home'].values[0] - team_info_df.loc[team_info_df['id'] == row['opponent_team'], 'strength_defence_away'].values[0]
            df.at[index, 'defence_strenght_difference'] = team_info_df.loc[team_info_df['name'] == row['team'], 'strength_defence_home'].values[0] - team_info_df.loc[team_info_df['id'] == row['opponent_team'], 'strength_attack_away'].values[0]

        else:
            df.at[index, 'attack_strenght_difference'] = team_info_df.loc[team_info_df['name'] == row['team'], 'strength_attack_away'].values[0] - team_info_df.loc[team_info_df['id'] == row['opponent_team'], 'strength_defence_home'].values[0]
            df.at[index, 'defence_strenght_difference'] = team_info_df.loc[team_info_df['name'] == row['team'], 'strength_defence_away'].values[0] - team_info_df.loc[team_info_df['id'] == row['opponent_team'], 'strength_attack_home'].values[0]

        df.at[index, 'strength_difference'] = team_info_df.loc[team_info_df['name'] == row['team'], 'strength'].values[0] - team_info_df.loc[team_info_df['id'] == row['opponent_team'], 'strength'].values[0]
    
    # Adding lagged features 
    NUM_LAGS = 5

    df.sort_values(by=['name', 'season', 'GW'], ascending=[True, True, True], inplace=True)
    
    # defining the features to lag
    features_not_to_lag = ['name', 'season', 'GW', 'position', 'team', 'opponent_team', 'was_home', 'attack_strenght_difference', 'defence_strenght_difference', 'strength_difference']
    lagged_features = list(set(relevant_features) - set(features_not_to_lag))

    lagged_columns = {}
    for feature in lagged_features:
        for i in range(1, NUM_LAGS + 1):
            # Create a new column with lagged values
            lagged_columns[f'{feature}_lag{i}'] = df.groupby('name')[feature].shift(i)

    # Combine the original DataFrame with the new lagged features at once
    lagged_df = pd.DataFrame(lagged_columns)
    df = pd.concat([df, lagged_df], axis=1)
    df.drop("season", axis=1, inplace=True)


    # A problem faced is that a lot of players are not playing, but they still affect the model. 
    # Assumption; I will make a classification model to remove most players that dont play the next match, thus I should not use them for the regression model analysis
    print("number of datapoints before filtering:", len(df))
    # Removing players that did not play in the last 2 GWs or red carded. IMPORTANT: This is a temporary solution, as I will later implement a classification model to remove players that do not play the next match
    df = df[((df['minutes'] > 0) | (df['minutes_lag1'] > 0)) & (df['red_cards_lag1'] == 0)]
    print("number of datapoints after filtering:", len(df))

    # Removing coloumns with info about the future
    items_to_keep = ['total_points', 'was_home'] # These are the only columns that should be used in the model, as they are not dependent on future information    
    coloumns_to_remove = list(set(relevant_features) - set(items_to_keep))
    
    df.drop(coloumns_to_remove, axis=1, inplace=True)

    return df
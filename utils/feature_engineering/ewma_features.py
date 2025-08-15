import pandas as pd
import numpy as np

def get_data_ewma(position: str, relevant_features: list[str]) -> pd.DataFrame:
    
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
    print(f"Num elements in {position} data before filter: {len(df)}")
    df = df[(df['minutes'] > 0)] # Also remove players with red cards
    print(f"Num elements in {position} data after filter: {len(df)}")


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
    
    df.sort_values(by=['name', 'season', 'GW'], ascending=[True, True, True], inplace=True)
    
    features_to_smooth = [
        'xP', 'assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 
        'expected_assists', 'expected_goal_involvements', 'expected_goals',
        'expected_goals_conceded', 'goals_scored', 'ict_index', 'influence', 
        'minutes', 'threat', 'value'
    ]

    # Error handling for missing columns
    features_to_smooth = [f for f in features_to_smooth if f in df.columns]

    # Calculate EWMA for each feature, grouped by player
    # span=5 is similar to a 5-week moving average, but gives more weight to recent games
    for feature in features_to_smooth:
        df[f'{feature}_ewma'] = df.groupby('name')[feature].transform(
            lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
        )

    # We keep the target and other non-lagged features
    ewma_features = [f'{f}_ewma' for f in features_to_smooth]
    strength_features = ['attack_strenght_difference', 'defence_strenght_difference', 'strength_difference']
    context_features = ['was_home', 'total_points'] # Keep target and our new weight
    
    features_to_keep = ewma_features + strength_features + context_features
    df = df[features_to_keep]
    df = df.fillna(0) # Fill any NaNs created by the initial EWMA steps

    return df
import pandas as pd

def fetch_gk_data_for_NGBoost(fetch_test_set: bool) -> pd.DataFrame:


    print(f"Creating hybrid feature set for GK")

    # Loading relevant data, this is done based on the feature_selection notebook

    gk_features = [
                'name', 'position', 'GW', 'xP', 'bps', 'clean_sheets',  
                'expected_goals_conceded', 'minutes', 
                'opponent_team', 'saves','total_points', 
                'value', 'was_home', 'team'                
                ]

    if fetch_test_set:
            df = pd.read_csv("../data/2024-25/gws/merged_gw.csv", usecols=gk_features)
            df['season'] = '2024-25'
             # An error in the data causes some rows to have opponent_team = 0 from round 22 and out
            df = df[df['opponent_team'] > 0]
    else:
        # Load and Merge Raw Data
        # Getting data from 22/23 season
        df_22_23 = pd.read_csv("../data/2022-23/gws/merged_gw.csv", usecols=gk_features)
        df_22_23['season'] = '2022-23'

        # Getting data from 23/24 season
        df_23_24 = pd.read_csv("../data/2023-24/gws/merged_gw.csv", usecols=gk_features)
        df_23_24['season'] = '2023-24'
        # Merging the data
        df = pd.concat([df_22_23, df_23_24], ignore_index=True)


    df = df[df['position'] == "GK"]
    print(f"Num elements in GK data before filter: {len(df)}")
    df = df[(df['minutes'] > 0)] # Also remove players with red cards
    print(f"Num elements in GK data after filter: {len(df)}")

    # Add Opponent Strength Features 
    team_info_cols = [
        'id', 'name', 'strength', 'strength_attack_home', 'strength_attack_away',
        'strength_defence_home', 'strength_defence_away'
    ]
    team_info_df_22_23 = pd.read_csv(f"../data/2022-23/teams.csv", usecols=team_info_cols)
    team_info_df_23_24 = pd.read_csv(f"../data/2023-24/teams.csv", usecols=team_info_cols)
    team_info_df_24_25 = pd.read_csv(f"../data/2024-25/teams.csv", usecols=team_info_cols)
    
    # This loop is slow; a merge would be faster but this is functionally correct
    for index, row in df.iterrows():
        team_info_df = team_info_df_22_23 if row['season'] == '2022-23' else team_info_df_23_24 if row['season'] == '2023-24' else team_info_df_24_25
        own_team_info = team_info_df[team_info_df['name'] == row['team']].iloc[0]
        opp_team_info = team_info_df[team_info_df['id'] == row['opponent_team']].iloc[0]

        if row['was_home']:
            df.at[index, 'attack_strength_difference'] = own_team_info['strength_attack_home'] - opp_team_info['strength_defence_away']
            df.at[index, 'defence_strength_difference'] = own_team_info['strength_defence_home'] - opp_team_info['strength_attack_away']
        else:
            df.at[index, 'attack_strength_difference'] = own_team_info['strength_attack_away'] - opp_team_info['strength_defence_home']
            df.at[index, 'defence_strength_difference'] = own_team_info['strength_defence_away'] - opp_team_info['strength_attack_home']
        df.at[index, 'strength_difference'] = own_team_info['strength'] - opp_team_info['strength']

    # Sort data chronologically for each player
    df.sort_values(by=['name', 'season', 'GW'], ascending=[True, True, True], inplace=True)

    # Engineer Features for Predicting the VARIANCE (scale) ---
    print("Creating features for predicting the variance (volatility, uncertainty)...")

    # Rolling Standard Deviation (Volatility)
    df['points_std_roll5'] = df.groupby('name')['total_points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).std()
    )
    # Haul & Blank Counts (Boom-or-Bust Metric)
    HAUL_THRESHOLD = 8
    BLANK_THRESHOLD = 3
    df['hauls_roll5'] = df.groupby('name')['total_points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).apply(lambda y: (y >= HAUL_THRESHOLD).sum())
    )
    df['blanks_roll5'] = df.groupby('name')['total_points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).apply(lambda y: (y < BLANK_THRESHOLD).sum())
    )

    


    # Create EWMA Features
    features_to_smooth = [
                          'clean_sheets', 'expected_goals_conceded', 
                          'saves','total_points',  
                        ]

    features_to_smooth = [f for f in features_to_smooth if f in df.columns]
    print("Creating EWMA features...")
    for feature in features_to_smooth:
        df[f'{feature}_ewma'] = df.groupby('name')[feature].transform(
            lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
        )


    lagged_features = [
                          'clean_sheets', 'expected_goals_conceded', 
                          'saves','total_points',  
                      ]
    NUM_LAGS = 2
    for feature in lagged_features:
        for i in range(1, NUM_LAGS + 1):
            # Create a new column with lagged values
            df[f'{feature}_lag{i}'] = df.groupby('name')[feature].shift(i)
    
    # Finalize Feature Set 
    ewma_cols = [f'{f}_ewma' for f in features_to_smooth]
    lag_cols = []
    for i in range(1, NUM_LAGS + 1):
        lag_cols.extend([f'{f}_lag{i}' for f in lagged_features])
    strength_cols = ['attack_strength_difference', 'defence_strength_difference', 'strength_difference']
    context_cols = ['was_home', 'total_points'] # Keep target variable
    variance_cols = ['points_std_roll5', 'hauls_roll5', 'blanks_roll5']

    features_to_keep = ewma_cols + lag_cols + strength_cols + context_cols + variance_cols
    df_final = df[features_to_keep].copy()
    df_final = df_final.fillna(0) 

    print("Improved Hybrid feature set created successfully")

    df_final['was_home'] = df_final['was_home'].astype(bool).astype(int)
    
    return df_final
    
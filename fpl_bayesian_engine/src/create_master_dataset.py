import pandas as pd
from pathlib import Path

def create_master_dataset(seasons: list, data_path: Path) -> pd.DataFrame:
    """
    Loads raw FPL data, engineers match-related and form features, 
    and returns a master DataFrame for modeling.

    Takes inspiration from the feature engineering logic in fwd.py.
    [cite: fwd.py]
    """
    print("Loading raw gameweek data...")
    
    # Define the core features needed from the raw files
    # [cite: fwd.py]
    features_to_load = [
        'name', 'team', 'minutes', 'ict_index', 
        'total_points', 'opponent_team', 'was_home', 
        'GW', 'position', 'value',
    ]

    # Load and combine data for all specified seasons
    all_gws = []
    for season in seasons:
        season_path = data_path / f"historic_data/{season}/gws/merged_gw.csv"
        try:
            df_season = pd.read_csv(season_path, usecols=features_to_load)
            df_season['season'] = season
            all_gws.append(df_season)

            #Getting only players playing in the current season
            df_player_names = pd.read_csv(data_path / "player_idlist.csv")
            df_player_names['name'] = df_player_names['first_name'] + " " + df_player_names['second_name']
            current_players = set(df_player_names['name'])

        except FileNotFoundError:
            print(f"Warning: Data for season {season} not found at {season_path}")
    
    if not all_gws:
        print("Error: No data loaded. Exiting.")
        return pd.DataFrame()

    df = pd.concat(all_gws, ignore_index=True)
    df = df[df['name'].isin(current_players)] # Filter out only players playing currently
    df = df[df['minutes'] > 0].copy() # Filter out players who didn't play

    # Engineer Match-Related Features 
    print("Engineering match-related strength features...")

    # Load team strength data for all seasons
    team_info_cols = [
        'id', 'name', 'strength', 'strength_attack_home', 'strength_attack_away',
        'strength_defence_home', 'strength_defence_away'
    ]
    team_info_df_22_23 = pd.read_csv(f"../data/historic_data/2022-23/teams.csv", usecols=team_info_cols)
    team_info_df_23_24 = pd.read_csv(f"../data/historic_data/2023-24/teams.csv", usecols=team_info_cols)
    team_info_df_24_25 = pd.read_csv(f"../data/historic_data/2024-25/teams.csv", usecols=team_info_cols)

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

    # Engineer Recent Form Features 
    print("Engineering recent form (EWMA) features...")
    
    # Sort data chronologically for each player to ensure correct EWMA calculation
    # [cite: fwd.py]
    df.sort_values(by=['name', 'season', 'GW'], ascending=[True, True, True], inplace=True)

    # Calculate EWMA features. shift(1) is used to prevent data leakage.
    # [cite: fwd.py]
    form_features = ['ict_index', 'minutes', 'total_points']
    for feature in form_features:
        df[f'{feature}_ewma'] = df.groupby('name')[feature].transform(
            lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
        )

    # Finalize Dataset 
    print("Finalizing master dataset...")
    
    final_features = [
        # Identifiers
        'name',
        'season',
        'GW',
        'position',
        'value',
        # Match-related features
        'strength_difference',
        'attack_strength_difference',
        'defence_strength_difference',
        'was_home',
        # Recent form features
        'ict_index_ewma',
        'minutes_ewma',
        'total_points_ewma',
        # Target variable
        'total_points'
    ]
    
    master_df = df[final_features].copy()
    
    # Fill NaN values that result from the shift operation (e.g., a player's first game)
    master_df.fillna(0, inplace=True)
    
    # Convert boolean 'was_home' to integer
    master_df['was_home'] = master_df['was_home'].astype(int)

    return master_df

if __name__ == "__main__":
    # Define the seasons you want to include in the dataset
    SEASONS_TO_PROCESS = ['2022-23', '2023-24', '2024-25']
    
    # Define the base path to your data directory
    # Assumes a structure like: your_project/data/SEASON/
    BASE_DATA_PATH = Path("../data")
    
    # Define the output file path
    OUTPUT_PATH = BASE_DATA_PATH / "processed" / "master_feature_dataset.csv"
    
    # Create the master dataset
    master_dataset = create_master_dataset(SEASONS_TO_PROCESS, BASE_DATA_PATH)
    
    if not master_dataset.empty:
        # Save the dataset to a CSV file
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        master_dataset.to_csv(OUTPUT_PATH, index=False)
        print(f"\nMaster dataset successfully created and saved to:\n{OUTPUT_PATH}")
        print(f"\nDataset shape: {master_dataset.shape}")
        print("\nFirst 5 rows:")
        print(master_dataset.head())
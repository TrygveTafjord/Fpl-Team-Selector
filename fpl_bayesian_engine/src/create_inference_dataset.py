import pandas as pd
from pathlib import Path

def create_inference_dataset(current_season: str, data_path: Path, start_gw: int, end_gw: int) -> pd.DataFrame:
    """
    Loads raw FPL data, engineers match-related and form features, 
    and returns a master DataFrame for modeling.
    start_gw = 1 and end_gw = 5 loads data from GW 1, 2, 3, 4, 5

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

    season_path = data_path / f"historic_data/{current_season}/gws/merged_gw.csv"
    try:
        df = pd.read_csv(season_path, usecols=features_to_load)
    except FileNotFoundError:
        print(f"Warning: Data for season {current_season} not found at {season_path}")

    # Only look at relevant features 
    df = df[(df['GW'] >= start_gw) & (df['GW'] <= end_gw)]

    # Engineer Match-Related Features 
    print("Engineering match-related strength features...")

    # Load team strength data for current season
    team_info_cols = [
        'id', 'name', 'strength', 'strength_attack_home', 'strength_attack_away',
        'strength_defence_home', 'strength_defence_away'
    ]
    team_info_df = pd.read_csv(f"../data/historic_data/{current_season}/teams.csv", usecols=team_info_cols)

    # This loop is slow; a merge would be faster but this is functionally correct
    for index, row in df.iterrows():
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
    df.sort_values(by=['name', 'GW'], ascending=[True, True], inplace=True)

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
    CURRENT_SEASON = '2025-26'
    
    FROM_GW = 1
    TO_GW = 3
    
    # Define the base path to your data directory
    # Assumes a structure like: your_project/data/SEASON/
    BASE_DATA_PATH = Path("../data")
    
    # Define the output file path
    OUTPUT_PATH = BASE_DATA_PATH / "processed" / f"inference_dataset_GW_{FROM_GW}_{TO_GW}.csv"
    
    # Create the master dataset
    master_dataset = create_inference_dataset(CURRENT_SEASON, BASE_DATA_PATH, FROM_GW, TO_GW)
    
    if not master_dataset.empty:
        # Save the dataset to a CSV file
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        master_dataset.to_csv(OUTPUT_PATH, index=False)
        print(f"\nMaster dataset successfully created and saved to:\n{OUTPUT_PATH}")
        print(f"\nDataset shape: {master_dataset.shape}")
        print("\nFirst 5 rows:")
        print(master_dataset.head())
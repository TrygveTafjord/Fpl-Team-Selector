import pandas as pd
import json
from pathlib import Path

def create_player_histogram_dict(data_path: Path) -> dict:

    print("Starting data loading and processing...")
    
    histogram_features = ['name', 'total_points', 'position', 'minutes']

    try:
        # Loading data from previous seasons 
        df_22_23 = pd.read_csv(data_path / "historic_data/2022-23/gws/merged_gw.csv", usecols=histogram_features)
        df_23_24 = pd.read_csv(data_path / "historic_data/2023-24/gws/merged_gw.csv", usecols=histogram_features)
        
        # Getting players for the upcoming season to filter the list 
        df_player_names = pd.read_csv(data_path / "player_idlist.csv")
        df_player_names['name'] = df_player_names['first_name'] + " " + df_player_names['second_name']
        current_players = set(df_player_names['name'])

    except FileNotFoundError as e:
        print(f"ERROR: Data file not found")
        print(f"Details: {e}")
        print("Please ensure the script is run from a directory where '../data' is a valid path.")
        return {}

    df_all_seasons = pd.concat([df_22_23, df_23_24], ignore_index=True)
    
    df_all_seasons = df_all_seasons[df_all_seasons['minutes'] > 0]

    # Filter for players active in the upcoming season 
    df_filtered = df_all_seasons[df_all_seasons['name'].isin(current_players)].copy()

    # Create the Points Occurrence Dictionary 
    point_counts = df_filtered.groupby(['name', 'total_points']).size()

    player_histogram_dict = {}
    for (name, points), count in point_counts.items():
        player_histogram_dict.setdefault(name, {})[int(points)] = int(count)
    
    print("Master dictionary created successfully.")
    return player_histogram_dict


def save_dict_to_json(data_dict: dict, output_path: Path):

    if not data_dict:
        print("Dictionary is empty. Skipping file save.")
        return

    print(f"Saving dictionary to {output_path}")
    
    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Use indent=4 for a human-readable "pretty-printed" file
        json.dump(data_dict, f, indent=4)
        
    print("Save complete!")

if __name__ == "__main__":

    BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    
    # Define where the final JSON file will be saved
    OUTPUT_FILE_PATH = BASE_DATA_DIR / "processed" / "player_points_histogram.json"

    # Creating the dictionary from the data
    master_histogram_dict = create_player_histogram_dict(BASE_DATA_DIR)
    
    # Saveing the resulting dictionary to the specified file
    save_dict_to_json(master_histogram_dict, OUTPUT_FILE_PATH)
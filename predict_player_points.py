import pandas as pd
import pickle
from utils.get_data import get_historical_player_data, get_upcoming_fixture_data, get_player_metadata
from pathlib import Path
import requests
# Constants for the columns in the fixture list

# Get the models
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
try:
    models = {}
    positions = ['fwd', 'mid', 'def', 'gk']
    for pos in positions:
        file_path = models_dir / f'ngboost_{pos}_model.pkl'
        with open(file_path, "rb") as f:
            models[pos.upper()] = pickle.load(f)

except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Get the FPL API data
fpl_api_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(fpl_api_url)
if response.status_code != 200:
    raise Exception(f"Failed to fetch data from {fpl_api_url}")
fpl_bootstrap_data = response.json()


CURRENT_GW = next(event['id'] for event in fpl_bootstrap_data['events'] if event['is_current'])

# Get data on the upcoming fixtures
NUM_FIXTURES_AHEAD = 5
fixture_list = get_upcoming_fixture_data(NUM_FIXTURES_AHEAD, fpl_bootstrap_data)

# Get relevant historical player data for each position
historic_data_by_position = {
    "GK": get_historical_player_data("GK"),
    "DEF": get_historical_player_data("DEF"),
    "MID": get_historical_player_data("MID"),
    "FWD": get_historical_player_data("FWD")
    }

# Get player metadata
player_metadata_df = get_player_metadata(fpl_bootstrap_data)
# Returns ['name', 'id', 'team', 'element_type', 'now_cost' 'chance_of_playing_this_round', 'chance_of_playing_next_round']

# Create a DataFrame to hold the results
results_df = player_metadata_df.copy()

# Creating empty columns to hold the future predictions
for i in range(1, NUM_FIXTURES_AHEAD + 1):
    results_df[f'predicted_points_{i}'] = 0.0
    results_df[f'predicted_points_distribution_{i}'] = None
    results_df[f'predicted_points_distribution_{i}'] = results_df[f'predicted_points_distribution_{i}'].astype(object)

for index, player in player_metadata_df.iterrows():

    # Skip players who are not expected to play this round
    if player["chance_of_playing_this_round"] == 0:
        continue
    
    position = player["position"]

    model = models.get(position) 
    if not model:
        raise ValueError(f"Invalid position, {position} at index {index}\n")

    historic_player_data = historic_data_by_position[position].loc[historic_data_by_position[position]['name'] == player["name"]]

    upcoming_fixtures = fixture_list[player["club"]]

    for i in range(NUM_FIXTURES_AHEAD):
        
        upcoming_fixture_data = upcoming_fixtures[i]

        # Check if we are at the end of the season
        if upcoming_fixture_data.empty:
            continue

        # Make predictions!
        X = pd.merge(historic_player_data, upcoming_fixture_data)
        X = X.drop(columns=["name"])
        point_prediction = model.predict(X)
        distribution_prediction = model.pred_dist(X)

        # Add the predictions to the result_df
        results_df.loc[index, f'predicted_points_{i+1}'] = point_prediction[0]
        results_df.loc[index, f'predicted_points_distribution_{i+1}'] = distribution_prediction

                     
results_df.to_csv(f"predictions/predictions_week_{CURRENT_GW}.csv", index=False)





    


    


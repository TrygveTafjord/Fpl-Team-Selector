import pandas as pd
from utils.get_data import get_last_five_matches_player_data, get_upcoming_fixture_data, get_player_metadata

# Constants for the columns in the fixture list

fwd_data_df = get_last_five_matches_player_data("FWD")
mid_data_df = get_last_five_matches_player_data("MID")
def_data_df = get_last_five_matches_player_data("DEF")
gk_data_df = get_last_five_matches_player_data("GK")    

player_metadata_df = get_player_metadata()

NUM_FIXTURES_AHEAD = 5
completed_player_predictions = []

fixture_list = get_upcoming_fixture_data(NUM_FIXTURES_AHEAD)

for player in player_metadata_df:
        
    position = player["position"]

    if position == "FWD":
        historic_player_data = fwd_data_df.loc[fwd_data_df['name'] == player["name"]]
    elif position == "MID":
        historic_player_data = mid_data_df.loc[mid_data_df['name'] == player["name"]]
    elif position == "DEF":
        historic_player_data = def_data_df.loc[def_data_df['name'] == player["name"]]
    elif position == "GK":
        historic_player_data = gk_data_df.loc[gk_data_df['name'] == player["name"]]
    else:
        raise ValueError("Invalid position")
    
    upcoming_fixtures = fixture_list[player["club"]]

    player_df = player.copy()

    for i in range(NUM_FIXTURES_AHEAD):
        
        fixture = pd.DataFrame(upcoming_fixtures[i], columns=["was_home", "strength_difference", "attack_strength_difference", "defense_strength_difference"])

        X = pd.merge(historic_player_data, fixture)
        X = X.drop(columns=["name"])
        
        #if position == "FWD":
        #    model = fwd_model
        #
        #elif position == "MID":
        #    model = mid_model
        #
        #elif position == "DEF":
        #    model = def_model
        #
        #elif position == "GK":
        #    model = gk_model

        #predicted_points = model.predict(X)

        #player_df[f"predicted_points_{i}_leg_ahead"] = predicted_points
    
    completed_player_predictions.append(player_df)

completed_player_predictions_df = pd.DataFrame(completed_player_predictions)

completed_player_predictions_df.to_csv("predictions.csv", index=False)





    


    


import pandas as pd
from utils.get_data import get_last_five_matches_player_data, get_upcoming_fixture_data, get_player_metadata, WAS_HOME, STRENGHT_DIFFERENCE, ATTACK_STRENGHT_DIFFERENCE, DEFENSE_STRENGHT_DIFFERENCE
from utils.teams import team_name_to_id

fwd_data_df = get_last_five_matches_player_data("FWD")
mid_data_df = get_last_five_matches_player_data("MID")
def_data_df = get_last_five_matches_player_data("DEF")
gk_data_df = get_last_five_matches_player_data("GK")    
player_metadata_df = get_player_metadata()

player_prediction_df = pd.DataFrame

NUM_FIXTURES_AHEAD = 5

upcoming_fixtures = get_upcoming_fixture_data(NUM_FIXTURES_AHEAD)



## iterate over players in player_metadata
## by using position, for each player, get the upcoming fixtures by checking the club id in the upcoming_fixtures dictionary
## by using position, get the correct lagged data dataframe for the player
## by using position, use the correct model to calculate the predicted points for the player
## add the metadata and predicted points to the player_prediction_df
## voila

    


    


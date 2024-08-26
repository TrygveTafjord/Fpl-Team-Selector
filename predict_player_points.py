import pandas as pd
from utils.get_data import get_last_five_matches_player_data, get_upcoming_fixture_data, WAS_HOME, STRENGHT_DIFFERENCE, ATTACK_STRENGHT_DIFFERENCE, DEFENSE_STRENGHT_DIFFERENCE
from utils.teams import team_name_to_id

fwd_data_df = get_last_five_matches_player_data("FWD")
mid_data_df = get_last_five_matches_player_data("MID")
def_data_df = get_last_five_matches_player_data("DEF")
gk_data_df = get_last_five_matches_player_data("GK")    

player_prediction_df = pd.DataFrame

NUM_FIXTURES_AHEAD = 5

upcoming_fixtures = get_upcoming_fixture_data(NUM_FIXTURES_AHEAD)

for fwd in fwd_data_df.iterrows():
        fixture_data = upcoming_fixtures[team_name_to_id[fwd['team_name']]]
        player_prediction_df['name', 'id', 'team', 'position'] = fwd['name', 'id', 'team', 'FWD']
        for fixture_nr in range(NUM_FIXTURES_AHEAD):
            fwd['was_home'] = fixture_data[fixture_nr][WAS_HOME]
            fwd['strength_difference'] = fixture_data[fixture_nr][STRENGHT_DIFFERENCE]
            fwd['attack_strength_difference'] = fixture_data[fixture_nr][ATTACK_STRENGHT_DIFFERENCE]
            #calculate predicted points for this fixture         

        fwd['was_home'] = fixture_data[fixture_nr][WAS_HOME]
        fwd['strength_difference'] = fixture_data[fixture_nr][STRENGHT_DIFFERENCE]
        fwd['attack_strength_difference'] = fixture_data[fixture_nr][ATTACK_STRENGHT_DIFFERENCE]
        #calculate predicted points for this fixture
        #add to player_prediction_df 


for fixture_nr in range(NUM_FIXTURES_AHEAD):
    for fwd in fwd_data_df.iterrows():
        fixture_data = upcoming_fixtures[team_name_to_id[fwd['team_name']]]
        fwd['was_home'] = fixture_data[fixture_nr][WAS_HOME]
        fwd['strength_difference'] = fixture_data[fixture_nr][STRENGHT_DIFFERENCE]
        fwd['attack_strength_difference'] = fixture_data[fixture_nr][ATTACK_STRENGHT_DIFFERENCE]
        #calculate predicted points for this fixture
        #add to player_prediction_df 

    for mid in mid_data_df.iterrows():
        fixture_data = upcoming_fixtures[team_name_to_id[mid['team_name']]]
        mid['was_home'] = fixture_data[fixture_nr][WAS_HOME]
        mid['strength_difference'] = fixture_data[fixture_nr][STRENGHT_DIFFERENCE]
        mid['attack_strength_difference'] = fixture_data[fixture_nr][ATTACK_STRENGHT_DIFFERENCE]
        #calculate predicted points for this fixture
        #add to player_prediction_df¨

    for defender in def_data_df.iterrows():
        fixture_data = upcoming_fixtures[team_name_to_id[defender['team_name']]]
        defender['was_home'] = fixture_data[fixture_nr][WAS_HOME]
        defender['strength_difference'] = fixture_data[fixture_nr][STRENGHT_DIFFERENCE]
        defender['defense_strength_difference'] = fixture_data[fixture_nr][ATTACK_STRENGHT_DIFFERENCE]
        #calculate predicted points for this fixture
        #add to player_prediction_df

    for gk in gk_data_df.iterrows():
        fixture_data = upcoming_fixtures[team_name_to_id[gk['team_name']]]
        gk['was_home'] = fixture_data[fixture_nr][WAS_HOME]
        gk['strength_difference'] = fixture_data[fixture_nr][STRENGHT_DIFFERENCE]
        gk['defense_strength_difference'] = fixture_data[fixture_nr][ATTACK_STRENGHT_DIFFERENCE]
        #calculate predicted points for this fixture
        #add to player_prediction_df

    




#for i in range(NUM_FIXTURES_AHEAD):
    


    


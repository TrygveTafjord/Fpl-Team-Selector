""" In data_exploration.ipynb I am doing feature exploratin to find out what features are relevant for a given position in the model. 
The results are stored here for the final model training. """


fpl_features_by_position = {
    "GK": [
                'name', 'position', 'GW', 'xP', 'bonus', 'bps', 'clean_sheets',  
                'expected_goals_conceded', 'ict_index', 'influence', 'minutes', 
                'opponent_team', 'own_goals', 'red_cards', 'saves',
                'selected', 'team_a_score', 'team_h_score', 'total_points', 
                'transfers_balance', 'transfers_in', 'transfers_out', 'value', 'was_home', 
                'yellow_cards', 
                ],
    "DEF": [
                'name', 'position', 'GW', 'xP', 'assists', 'bonus', 'bps', 'clean_sheets', 
                'creativity', 'expected_assists', 'expected_goal_involvements', 
                'expected_goals','expected_goals_conceded', 'goals_scored', 'ict_index', 'influence', 'minutes', 
                'opponent_team', 'own_goals', 'penalties_missed', 'red_cards', 
                'selected', 'team_a_score', 'team_h_score', 'threat', 'total_points', 
                'transfers_balance', 'transfers_in', 'transfers_out', 'value', 'was_home', 
                'yellow_cards'
                ],
    "MID": [
                'name', 'position', 'GW', 'xP', 'assists', 'bonus', 'bps', 'clean_sheets', 
                'creativity', 'expected_assists', 'expected_goal_involvements', 
                'expected_goals','expected_goals_conceded', 'goals_scored', 'ict_index', 'influence', 'minutes', 
                'opponent_team', 'own_goals', 'penalties_missed', 'red_cards', 
                'selected', 'team_a_score', 'team_h_score', 'threat', 'total_points', 
                'transfers_balance', 'transfers_in', 'transfers_out', 'value', 'was_home', 
                'yellow_cards'
                ],
    "FWD": [
                'name', 'position', 'GW', 'xP', 'assists', 'bonus', 'bps', 
                'creativity', 'expected_assists', 'expected_goal_involvements', 
                'expected_goals', 'goals_scored', 'ict_index', 'influence', 'minutes', 
                'opponent_team', 'own_goals', 'penalties_missed', 'selected', 
                'team_a_score', 'team_h_score', 'threat', 'total_points', 
                'transfers_balance', 'transfers_in', 'transfers_out', 'value', 'was_home', 
                'yellow_cards'
                ]
}



""" In data_exploration.ipynb I am doing feature exploratin to find out what features are relevant for a given position in the model. 
The results are stored here for the final model training. """


fpl_features_by_position = {
    "GK":   [
            'name', 'position', 'GW', 'xP', 'bps', 'clean_sheets',  
            'expected_goals_conceded', 'minutes', 
            'opponent_team', 'saves','total_points', 
            'value', 'was_home', 'team'                
            ],
        
    "DEF":  [
            'name', 'position', 'team', 'xP', 'assists', 'bps', 'clean_sheets', 
            'expected_assists', 'expected_goal_involvements', 
            'expected_goals_conceded', 'influence', 'minutes', 
            'opponent_team', 'own_goals', 'selected', 'team_a_score',
            'team_h_score', 'total_points','value', 'was_home', 'yellow_cards', 'GW'
            ],
        
    "MID":  [
            'name', 'position', 'team', 'xP', 'assists', 'bps', 'clean_sheets', 
            'expected_assists', 'creativity', 'expected_assists', 'expected_goal_involvements', 
            'expected_goals', 'goals_scored', 'ict_index', 'influence', 'minutes', 
            'opponent_team', 'selected', 'threat', 'total_points', 'value', 'was_home', 
            'GW'
            ],
    
    "FWD":  [
            'name', 'position', 'team', 'xP', 'assists', 'bps', 
            'creativity', 'expected_assists', 'expected_goal_involvements', 
            'expected_goals', 'goals_scored', 'ict_index', 'influence', 'minutes', 
            'opponent_team', 'selected', 'threat', 'total_points', 'value', 'was_home', 
            'GW'
            ],
            
}


fpl_lagged_features_by_position = {
    "GK":       [
                 'clean_sheets', 'expected_goals_conceded', 
                 'saves','total_points',  
                ],

    "DEF":      [ 
                'bps', 'clean_sheets', 'expected_goal_involvements', 
                'expected_goals_conceded', 'minutes', 'total_points', 'value', 
                'yellow_cards'
                ],

    "MID":      [
                'bps', 'expected_assists', 'expected_goals', 'goals_scored', 'ict_index',
                'influence', 'minutes', 'threat', 'total_points',  
                ],

    "FWD":      [
                'bps', 'expected_assists', 'expected_goals', 'goals_scored', 'ict_index',
                'influence', 'minutes', 'threat', 'total_points',  
                ]
}

fpl_features_to_smooth_by_position = {
    "GK":       [
                 'clean_sheets', 'expected_goals_conceded', 
                 'saves','total_points',  
                ],

    "DEF":      [
                'bps', 'clean_sheets', 
                'expected_goal_involvements','expected_goals_conceded', 
                'influence','minutes', 'yellow_cards', 'total_points'
                ],

    "MID":      [
                'bps', 'creativity', 'expected_assists', 'expected_goal_involvements', 
                'clean_sheets', 'expected_goals', 'ict_index', 'minutes', 
                'total_points', 
                ],
                
    "FWD":      [
                'bps', 'creativity', 'expected_assists', 'expected_goal_involvements', 
                'expected_goals', 'ict_index', 'minutes', 'total_points', 
                ]
}

import pandas as pd
import os
from enum import Enum

class Season(str, Enum):
    SEASON_2022_23 = "2022-23" ,
    SEASON_2023_24 = "2023-24"


def get_player_history_by_season(season: Season):
    directory = f'./data/{season}/players'
    dfs = []
    for player in os.listdir(directory):
        df = pd.read_csv(f'{directory}/{player}/gw.csv')
        dfs.append(df)
    return pd.concat(dfs)

# fpl_team_selector
####Making a point-predictor and team selector for fantasy premier league
points are predicted using player-data from the five previously played matches, as well as information about upcoming fixtures.
A random forrest model is used, a seperate model is made for each position (GK, DEF, MID, FWD).


#### Historic data is fetched form the following repo:
https://github.com/vaastav/Fantasy-Premier-League

#### Data on players and upcoming fixtures is fetched from the fpl-api
Information on how to use the api is gotten from:
https://medium.com/@frenzelts/fantasy-premier-league-api-endpoints-a-detailed-guide-acbd5598eb19

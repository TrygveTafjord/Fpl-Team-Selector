# Fantasy Premier League Point Predictor

This project aims to create a point predictor and team selector for Fantasy Premier League (FPL). The predictor uses a Random Forest model to predict points based on player data from the five most recent matches and upcoming fixture information. Separate models are created for each player position (GK, DEF, MID, FWD).

## Data Sources

### Historical Player Data

Historical data for player performances is fetched from a publicly available dataset. The data includes various metrics collected over multiple seasons, providing the foundation for our model's predictions.

**Citation:**
```bibtex
@misc{anand2016fantasypremierleague,
  title = {{FPL Historical Dataset}},
  author = {Anand, Vaastav},
  year = {2022},
  howpublished = {Retrieved August 2022 from \url{https://github.com/vaastav/Fantasy-Premier-League/}}
}
```

### Real-Time Player and Fixture Data
Real-time data on players and upcoming fixtures is retrieved from the official Fantasy Premier League API. This data is used to inform the predictions for upcoming gameweeks and injuries.

Information on how to interact with the FPL API was obtained from this detailed guide: [Fantasy Premier League API Endpoints Guide](https://medium.com/@frenzelts/fantasy-premier-league-api-endpoints-a-detailed-guide-acbd5598eb19).

### Dependencies
Python (>= 3.7)
pandas
matplotlib
scikit-learn
requests

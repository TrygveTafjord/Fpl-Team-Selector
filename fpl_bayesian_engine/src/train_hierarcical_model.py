import pandas as pd
import pymc as pm
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

def prepare_full_dataset(master_df_path: Path, player_idlist_path: Path) -> Dict[str, Any]:
    """
    Loads master feature data and the current player list, then prepares it for the hierarchical model.

    This updated function:
    1.  Loads the definitive list of current players from player_idlist.csv to establish the "universe".
    2.  Creates price tiers and mappings for ALL current players, including new ones.
    3.  Loads the historical master dataset and maps its data onto the established player universe.
    4.  Generates unique integer indices for all players and position-price tiers.
    5.  Scales the historical features using StandardScaler.
    6.  Returns a dictionary containing all necessary data for the PyMC model.
    """
    print("Preparing full dataset for hierarchical modeling...\n")

    # Load the definitive list of ALL current players 
    try:
        df_players = pd.read_csv(player_idlist_path)

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure player_idlist.csv path is correct.")
        raise
    
    print(f"Loaded {len(df_players)} players for the current season from {player_idlist_path.name}.")

    # Create Price Tiers for ALL current players (value is given in order of 10s in dataset)
    print("Creating price tiers for all current players...\n")
    df_players['price'] = df_players['value'] / 10.0
    
    # Define price bins for each position.
    fwd_bins = [0, 6.5, 8.0, 9.5, 15.0]
    mid_bins = [0, 6.5, 8.0, 9.5, 15.0]
    def_bins = [0, 4.5, 5.5, 6.5, 10.0]
    gkp_bins = [0, 4.5, 5.0, 5.5, 10.0]
    # Standardized labels for the tiers
    labels = ['Budget', 'Mid', 'Good', 'Premium']
    
    # Apply bins based on position to the complete player dataframe
    df_players.loc[df_players['position'] == 'FWD', 'price_tier'] = pd.cut(df_players['price'], bins=fwd_bins, labels=labels, right=False)
    df_players.loc[df_players['position'] == 'MID', 'price_tier'] = pd.cut(df_players['price'], bins=mid_bins, labels=labels, right=False)
    df_players.loc[df_players['position'] == 'DEF', 'price_tier'] = pd.cut(df_players['price'], bins=def_bins, labels=labels, right=False)
    df_players.loc[df_players['position'] == 'GK', 'price_tier'] = pd.cut(df_players['price'], bins=gkp_bins, labels=labels, right=False) 

    df_players['price_tier'] = df_players['price_tier'].cat.add_categories('Other').fillna('Other')
    df_players['pos_price_tier'] = df_players['position'] + '_' + df_players['price_tier'].astype(str)

    # Create the master mappings from the complete player list 
    # These maps define the coordinates of our PyMC model.
    player_map = pd.Index(df_players['name'].unique())
    pos_price_map = pd.Index(df_players['pos_price_tier'].unique())
    
    # Create dictionary for fast lookups
    player_to_idx = {name: i for i, name in enumerate(player_map)}
    pos_price_to_idx = {tier: i for i, tier in enumerate(pos_price_map)}
    
    # Map these indices back to the main player dataframe
    df_players['player_idx'] = df_players['name'].map(player_to_idx)
    df_players['pos_price_idx'] = df_players['pos_price_tier'].map(pos_price_to_idx)
    
    print(f"Created mappings for {len(player_map)} players and {len(pos_price_map)} position-price tiers.")

    # Load historical data and map it to our universe, so see how the dataset was made, see create_master_dataset.py
    print("Loading historical gameweek data...")
    try:
        df_hist = pd.read_csv(master_df_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure your master feature dataset path is correct.")
        raise

    print(f"Initial length of historical dataset: {len(df_hist)}")
    
    # Filter historical data to only include players that are in our current player list
    df_hist = df_hist[df_hist['name'].isin(player_map)].copy()
    
    original_rows = len(df_hist)

    #We need to remove players with negative points because we have assumed a negative binomial distribution that is strictly positive
    #Otherwise the code breaks
    df_hist = df_hist[df_hist['total_points'] >= 0]
    print(f"Removed {original_rows - len(df_hist)} rows belonging to players with < 0 points.")
    
    if df_hist.empty:
        raise ValueError("No historical data found for any of the players in player_idlist.csv. Check for name mismatches.")

    # Map historical data to the predefined indices from the current player list
    df_hist['player_idx'] = df_hist['name'].map(player_to_idx)
    
    # We need to determine the tier for historical games as well, to link to the correct tier intercept
    df_hist['price'] = df_hist['value'] / 10.0
    df_hist.loc[df_hist['position'] == 'FWD', 'price_tier'] = pd.cut(df_hist['price'], bins=fwd_bins, labels=labels, right=False)
    df_hist.loc[df_hist['position'] == 'MID', 'price_tier'] = pd.cut(df_hist['price'], bins=mid_bins, labels=labels, right=False)
    df_hist.loc[df_hist['position'] == 'DEF', 'price_tier'] = pd.cut(df_hist['price'], bins=def_bins, labels=labels, right=False)
    df_hist.loc[df_hist['position'] == 'GK', 'price_tier'] = pd.cut(df_hist['price'], bins=gkp_bins, labels=labels, right=False)
    df_hist['price_tier'] = df_hist['price_tier'].cat.add_categories('Other').fillna('Other')
    df_hist['pos_price_tier'] = df_hist['position'] + '_' + df_hist['price_tier'].astype(str)
    df_hist['pos_price_idx'] = df_hist['pos_price_tier'].map(pos_price_to_idx)

    # Drop rows where a historical tier might not exist in the current season's tiers (edge case)
    df_hist.dropna(subset=['player_idx', 'pos_price_idx'], inplace=True)
    df_hist['player_idx'] = df_hist['player_idx'].astype(int)
    df_hist['pos_price_idx'] = df_hist['pos_price_idx'].astype(int)

    # Define features and scale them based on historical data
    features = [
        'strength_difference', 'attack_strength_difference', 
        'defence_strength_difference', 'was_home',
        'ict_index_ewma', 'minutes_ewma', 'total_points_ewma'
    ]
    X = df_hist[features]
    y = df_hist['total_points']
    
    #Scaling is needed to ensure convergation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Dataset preparation complete.")
    print(f"Final length of historical training data: {len(df_hist)}")

    return {
        "X_scaled": X_scaled,
        "y": y.values,
        "pos_price_idx_data": df_hist['pos_price_idx'].values, # For linking historical data to tiers
        "player_idx_data": df_hist['player_idx'].values, # For linking historical data to players
        "player_pos_price_idx_map": df_players[['player_idx', 'pos_price_idx']].set_index('player_idx')['pos_price_idx'].values,
        "n_players": len(player_map),
        "n_pos_price_tiers": len(pos_price_map),
        "n_features": len(features),
        "feature_names": features,
        "player_map": player_map,
        "pos_price_map": pos_price_map,
        "scaler": scaler
    }

def define_price_hierarchical_model(data: Dict[str, Any]) -> pm.Model:
    """
    Defines the hierarchical bimodal Negative Binomial model in PyMC.
    
    The hierarchy is: Global -> Position-Price Tier -> Player
    """
    coords = {
        "feature": data["feature_names"],
        "pos_price_tier": data["pos_price_map"],
        "player": data["player_map"]
    }
    with pm.Model(coords=coords) as hierarchical_model:
        
        # --- Hyperpriors (Global Level) ---
        global_w_intercept = pm.Normal('global_w_intercept', mu=-1.5, sigma=1.0)
        global_blank_intercept = pm.Normal('global_blank_intercept', mu=np.log(2.5), sigma=0.5)
        global_haul_offset_intercept = pm.Normal('global_haul_offset_intercept', mu=np.log(7.0), sigma=0.5)

        # --- Position-Price Tier Level Priors ---
        tier_w_intercept = pm.Normal('tier_w_intercept', mu=global_w_intercept, sigma=1.0, dims="pos_price_tier") 
        tier_blank_intercept = pm.Normal('tier_blank_intercept', mu=global_blank_intercept, sigma=0.75, dims="pos_price_tier") 
        tier_haul_offset_intercept = pm.Normal('tier_haul_offset_intercept', mu=global_haul_offset_intercept, sigma=0.75, dims="pos_price_tier") 
        
        # --- Player-Level Offsets (defined for ALL players) ---
        player_w_offset = pm.Normal('player_w_offset', mu=0, sigma=0.8, dims="player") 
        player_blank_offset = pm.Normal('player_blank_offset', mu=0, sigma=0.7, dims="player") 
        player_haul_offset = pm.Normal('player_haul_offset', mu=0, sigma=0.7, dims="player") 

        # Map each player to their tier's intercept
        # data["player_pos_price_idx_map"] is an array where the index is the player_idx and the value is the pos_price_idx
        player_tier_w_intercepts = tier_w_intercept[data["player_pos_price_idx_map"]]
        player_tier_blank_intercepts = tier_blank_intercept[data["player_pos_price_idx_map"]]
        player_tier_haul_intercepts = tier_haul_offset_intercept[data["player_pos_price_idx_map"]]

        # Construct the full player-level intercepts by adding player offsets to their respective tier means
        # These are now defined for all players, not just those in the data
        player_w_intercept_full = player_tier_w_intercepts + player_w_offset
        player_blank_intercept_full = player_tier_blank_intercepts + player_blank_offset
        player_haul_offset_intercept_full = player_tier_haul_intercepts + player_haul_offset

        # --- Coefficients & Dispersion (Global) ---
        beta_w_coeffs = pm.Normal('beta_w_coeffs', mu=0, sigma=1.0, dims="feature") 
        beta_blank_coeffs = pm.Normal('beta_blank_coeffs', mu=0, sigma=1.0, dims="feature") 
        beta_offset_coeffs = pm.Normal('beta_offset_coeffs', mu=0, sigma=1.0, dims="feature") 

        alpha_blank = pm.HalfNormal('alpha_blank', sigma=1.0)
        alpha_haul = pm.HalfNormal('alpha_haul', sigma=2.0)
        
        epsilon = 1e-6
        mu_max = 40.0

        # --- Link Functions & Likelihood (applied ONLY to the training data) ---
        # Select the intercepts corresponding to the players in the historical data
        w_logit = player_w_intercept_full[data["player_idx_data"]] + pm.math.dot(data["X_scaled"], beta_w_coeffs)
        w_unclipped = pm.math.sigmoid(w_logit)
        w = pm.math.clip(w_unclipped, epsilon, 1.0 - epsilon)

        mu_blank_log = player_blank_intercept_full[data["player_idx_data"]] + pm.math.dot(data["X_scaled"], beta_blank_coeffs)
        mu_blank_unbounded = pm.math.exp(mu_blank_log)
        mu_blank = pm.math.clip(mu_blank_unbounded, epsilon, mu_max)
        
        mu_haul_offset_log = player_haul_offset_intercept_full[data["player_idx_data"]] + pm.math.dot(data["X_scaled"], beta_offset_coeffs)
        mu_haul = mu_blank + pm.math.exp(mu_haul_offset_log) 

        nb_blank = pm.NegativeBinomial.dist(mu=mu_blank, alpha=alpha_blank)
        nb_haul = pm.NegativeBinomial.dist(mu=mu_haul, alpha=alpha_haul)
        
        weights = pm.math.stack([1.0 - w, w], axis=1)

        likelihood = pm.Mixture(
            'likelihood', 
            w=weights, 
            comp_dists=[nb_blank, nb_haul],
            observed=data["y"]
        )        
    return hierarchical_model


if __name__ == "__main__":
    # Define file paths
    BASE_DATA_PATH = Path("../data")
    MASTER_DF_PATH = BASE_DATA_PATH / "processed" / "master_feature_dataset.csv"
    PLAYER_IDLIST_PATH = BASE_DATA_PATH / "player_idlist.csv" # Path to the definitive player list
    
    MODEL_OUTPUT_PATH = Path("../models/")
    MODEL_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    # Prepare data using the new logic
    data_dict = prepare_full_dataset(MASTER_DF_PATH, PLAYER_IDLIST_PATH)
    
    # Define the model
    hierarchical_model = define_price_hierarchical_model(data_dict)

    # Train the model
    print("\nStarting MCMC sampling... This may take a while.")
    with hierarchical_model:
        idata = pm.sample(2000, tune=1000, chains=4, cores=-1, target_accept=0.9, init="advi+adapt_diag")
    
    # Save the trained model and artifacts
    print("Sampling complete. Saving model and artifacts...")
    model_file = MODEL_OUTPUT_PATH / "hierarchical_bimodal_model.nc"
    artifacts_file = MODEL_OUTPUT_PATH / "model_artifacts.pkl"

    idata.to_netcdf(model_file)
    
    # The saved artifacts now contain mappings for ALL players
    with open(artifacts_file, "wb") as f:
        pickle.dump({
            "scaler": data_dict["scaler"],
            "player_map": data_dict["player_map"],
            "pos_price_map": data_dict["pos_price_map"],
            "feature_names": data_dict["feature_names"],
            "player_pos_price_idx_map": data_dict["player_pos_price_idx_map"]
        }, f)
        
    print(f"\nModel saved to: {model_file}")
    print(f"Artifacts saved to: {artifacts_file}")
    print("\nTraining process finished successfully.")
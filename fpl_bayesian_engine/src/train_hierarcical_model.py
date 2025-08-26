import pandas as pd
import pymc as pm
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

def prepare_full_dataset(master_df_path: Path) -> Dict[str, Any]:
    """
    Loads master feature data and the current player list, then prepares it for the hierarchical model.

    This function:
    - Filters the historical master dataset to only include players in the current player_idlist.csv.
    - Uses the 'value' column from the master dataset to create price tiers.
    - Generates unique integer indices for players and position-price tiers.
    - Scales the features using StandardScaler.
    - Returns a dictionary containing all necessary data for the PyMC model.
    """
    print("Preparing full dataset for hierarchical modeling...")

    # Load the master dataset of all historical gameweeks
    try:
        df = pd.read_csv(master_df_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure your file paths are correct.")
        raise
    print(f"Initial lenght of dataset {len(df)}")

    # Filter out players with under 5 observations
    print("Filtering out players with insufficient data...")
    player_counts = df['name'].value_counts()
    players_to_keep = player_counts[player_counts >= 5].index.tolist()

    # Filter out players with negative points, the negative binomal model is strictly positive
    original_rows = len(df)
    df = df[df['name'].isin(players_to_keep)].copy()
    print(f"Removed {original_rows - len(df)} rows belonging to {len(player_counts) - len(players_to_keep)} players with < 5 observations.") 
    
    original_rows = len(df)
    df = df[df['total_points'] >= 0]
    print(f"Removed {original_rows - len(df)} rows belonging to players with < 0 points.")

    if df.empty:
        raise ValueError("No players from player_idlist.csv were found in the master_feature_dataset.csv. Check for name mismatches.")

    # Create Price Tiers using the 'value' column from the master dataset 
    print("Creating price tiers...")

    # The 'value' column is price * 10 (ex. 65 for £6.5m)
    df['price'] = df['value'] / 10.0
    
    # Define price bins for each position. These can be tuned.
    fwd_bins = [0, 6.5, 8.0, 9.5, 15.0]
    mid_bins = [0, 6.5, 8.0, 9.5, 15.0]
    def_bins = [0, 4.5, 5.5, 6.5, 10.0]
    gkp_bins = [0, 4.5, 5.0, 5.5, 10.0]
    labels = ['Budget', 'Mid', 'Good', 'Premium']
    
    # Apply bins based on position. A 'position' column must exist in your master_feature_dataset.csv
    if 'position' not in df.columns:
        raise KeyError("The 'position' column was not found in master_feature_dataset.csv. Please ensure it is included.")
        
    df.loc[df['position'] == 'FWD', 'price_tier'] = pd.cut(df['price'], bins=fwd_bins, labels=labels, right=False)
    df.loc[df['position'] == 'MID', 'price_tier'] = pd.cut(df['price'], bins=mid_bins, labels=labels, right=False)
    df.loc[df['position'] == 'DEF', 'price_tier'] = pd.cut(df['price'], bins=def_bins, labels=labels, right=False)
    df.loc[df['position'] == 'GKP', 'price_tier'] = pd.cut(df['price'], bins=gkp_bins, labels=labels, right=False)

    df['price_tier'] = df['price_tier'].cat.add_categories('Other').fillna('Other')

    # Create a unique index for the position-price tier group
    df['pos_price_tier'] = df['position'] + '_' + df['price_tier'].astype(str)
    
    # Create integer codes for the hierarchical levels using "name" as the unique player identifier
    df['pos_price_idx'], pos_price_map = pd.factorize(df['pos_price_tier'])
    df['player_idx'], player_map = pd.factorize(df['name'])
    
    # Define features and scale them
    features = [
        'strength_difference', 'attack_strength_difference', 
        'defence_strength_difference', 'was_home',
        'ict_index_ewma', 'minutes_ewma', 'total_points_ewma'
    ]
    X = df[features]
    y = df['total_points']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Dataset preparation complete. \n")
    print(f"final lenght of dataset {len(df)}")

    return {
        "X_scaled": X_scaled,
        "y": y.values,
        "pos_price_idx": df['pos_price_idx'].values,
        "player_idx": df['player_idx'].values,
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
        
        # Hyperpriors (Global Level - Priors for the Priors) 
        global_w_intercept = pm.Normal('global_w_intercept', mu=-1.5, sigma=1.0)
        global_blank_intercept = pm.Normal('global_blank_intercept', mu=np.log(2.5), sigma=0.5)
        global_haul_offset_intercept = pm.Normal('global_haul_offset_intercept', mu=np.log(7.0), sigma=0.5)

        # Position-Price Tier Level Priors 
        tier_w_intercept = pm.Normal('tier_w_intercept', mu=global_w_intercept, sigma=1.0, dims="pos_price_tier") 
        tier_blank_intercept = pm.Normal('tier_blank_intercept', mu=global_blank_intercept, sigma=0.75, dims="pos_price_tier") 
        tier_haul_offset_intercept = pm.Normal('tier_haul_offset_intercept', mu=global_haul_offset_intercept, sigma=0.75, dims="pos_price_tier") 
        
        # Player-Level Offsets from the Tier Mean (one offset per player)
        player_w_offset = pm.Normal('player_w_offset', mu=0, sigma=0.8, dims="player") 
        player_blank_offset = pm.Normal('player_blank_offset', mu=0, sigma=0.7, dims="player") 
        player_haul_offset = pm.Normal('player_haul_offset', mu=0, sigma=0.7, dims="player") 

        # Construct the full player-level parameters by adding player offsets to their respective tier means
        player_w_intercept = tier_w_intercept[data["pos_price_idx"]] + player_w_offset[data["player_idx"]]
        player_blank_intercept = tier_blank_intercept[data["pos_price_idx"]] + player_blank_offset[data["player_idx"]]
        player_haul_offset_intercept = tier_haul_offset_intercept[data["pos_price_idx"]] + player_haul_offset[data["player_idx"]]
        
        # Coefficients & Dispersion (Global) 
        beta_w_coeffs = pm.Normal('beta_w_coeffs', mu=0, sigma=1.0, dims="feature") 
        beta_blank_coeffs = pm.Normal('beta_blank_coeffs', mu=0, sigma=1.0, dims="feature") 
        beta_offset_coeffs = pm.Normal('beta_offset_coeffs', mu=0, sigma=1.0, dims="feature") 

        alpha_blank = pm.HalfNormal('alpha_blank', sigma=1.0)
        alpha_haul = pm.HalfNormal('alpha_haul', sigma=2.0)
        
        # Define stability constants
        epsilon = 1e-6
        mu_max = 40.0

        # Link Functions & Likelihood 
        w_logit = player_w_intercept + pm.math.dot(data["X_scaled"], beta_w_coeffs)
        w_unclipped = pm.math.sigmoid(w_logit)
        w = pm.math.clip(w_unclipped, epsilon, 1.0 - epsilon)

        mu_blank_log = player_blank_intercept + pm.math.dot(data["X_scaled"], beta_blank_coeffs)
        mu_blank_unbounded = pm.math.exp(mu_blank_log)
        mu_blank = pm.math.clip(mu_blank_unbounded, epsilon, mu_max)
        
        mu_haul_offset_log = player_haul_offset_intercept + pm.math.dot(data["X_scaled"], beta_offset_coeffs)
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
    MASTER_DF_PATH = Path("../data/processed/master_feature_dataset.csv")
    
    MODEL_OUTPUT_PATH = Path("../models/")
    MODEL_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    # Prepare data
    data_dict = prepare_full_dataset(MASTER_DF_PATH)
    
    # Define the model
    hierarchical_model = define_price_hierarchical_model(data_dict)

    # Train the model (this is the one and only inference step)
    print("\nStarting MCMC sampling... This may take a while.")
    with hierarchical_model:
        idata = pm.sample(2000, tune=1000, chains=4, cores=7, target_accept=0.9, init="advi+adapt_diag")
    
    # Save the trained model and artifacts
    print("Sampling complete. Saving model and artifacts...")
    model_file = MODEL_OUTPUT_PATH / "hierarchical_bimodal_model.nc"
    artifacts_file = MODEL_OUTPUT_PATH / "model_artifacts.pkl"

    idata.to_netcdf(model_file)
    
    with open(artifacts_file, "wb") as f:
        pickle.dump({
            "scaler": data_dict["scaler"],
            "player_map": data_dict["player_map"],
            "pos_price_map": data_dict["pos_price_map"],
            "feature_names": data_dict["feature_names"]
        }, f)
        
    print(f"\nModel saved to: {model_file}")
    print(f"Artifacts saved to: {artifacts_file}")
    print("\nTraining process finished successfully.")
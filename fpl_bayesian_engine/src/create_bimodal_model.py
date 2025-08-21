import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pytensor
from pathlib import Path
from sklearn.preprocessing import StandardScaler


pytensor.config.mode == 'NUMBA'

def prepare_player_data(master_df: pd.DataFrame, player_name: str):

    print(f"\n--- Step 2: Preparing data for {player_name} ---")
    
    player_df = master_df[master_df['name'] == player_name].copy()
    
    if player_df.empty:
        raise ValueError(f"No data found for player: {player_name}")

    features = [
        'strength_difference', 
        'attack_strength_difference', 
        'defence_strength_difference', 
        'was_home',
        'ict_index_ewma',  
        'minutes_ewma', 
        'total_points_ewma'
    ]
    
    X_unscaled = player_df[features]
    y = player_df['total_points']

    # scale the features 
    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(X_unscaled)
    X = pd.DataFrame(X_scaled_values, columns=features, index=X_unscaled.index)
    
    print(f"Data prepared and scaled. Found {len(player_df)} historical matches.")

    return X, y, features


def define_bimodal_model(X: pd.DataFrame, y: pd.Series, features: list):

    print(f"\n--- Step 3: Defining the Bayesian Bimodal Model ---")
    
    with pm.Model() as bimodal_model:
        # --- Priors for the regression coefficients (betas) ---
        # We need separate coefficients for each component of the mixture
        
        # In define_bimodal_model function...

        # 1. Priors for the "Haul" probability (w) regression
        beta_w_intercept = pm.Normal('beta_w_intercept', mu=0.5, sigma=0.5)
        beta_w_coeffs = pm.Normal('beta_w_coeffs', mu=0.0, sigma=0.5, shape=len(features))

        # 2. Priors for the "Blank" component (mu1) regression
        beta_blank_intercept = pm.Normal('beta_blank_intercept', mu=np.log(2), sigma=1.0) 
        beta_blank_coeffs = pm.Normal('beta_blank_coeffs', mu=0.0, sigma=1.0, shape=len(features))

        # 3. Priors for the "Haul" component (mu2) regression
        beta_haul_intercept = pm.Normal('beta_haul_intercept', mu=np.log(8), sigma=1.0)
        beta_haul_coeffs = pm.Normal('beta_haul_coeffs', mu=0.0, sigma=1.0, shape=len(features))

        # Priors for the dispersion parameters (alpha) 
        alpha_blank = pm.Exponential('alpha_blank', 1.0)
        alpha_haul = pm.Exponential('alpha_haul', 1.0)

        # Link Functions (Regression Equations) 
        # These equations connect the features to the distribution parameters
        
        # Equation for the probability of a haul (w)
        w_logit = beta_w_intercept + pm.math.dot(X.values, beta_w_coeffs)
        w = pm.Deterministic('w', pm.math.sigmoid(w_logit)) # Sigmoid transforms to a probability (0-1)
        
        # Equation for the mean of the "Blank" component (mu1)
        mu_blank_log = beta_blank_intercept + pm.math.dot(X.values, beta_blank_coeffs)
        mu_blank = pm.Deterministic('mu_blank', pm.math.exp(mu_blank_log)) # Exp transforms back from log scale
        
        # Equation for the mean of the "Haul" component (mu2)
        mu_haul_log = beta_haul_intercept + pm.math.dot(X.values, beta_haul_coeffs)
        mu_haul = pm.Deterministic('mu_haul', pm.math.exp(mu_haul_log))

        # Likelihood Function 
        # This is the Negative Bimodal distribution that generates the observed points
        # It's a mixture of two Negative Binomial distributions
        
        nb_blank = pm.NegativeBinomial.dist(mu=mu_blank, alpha=alpha_blank)
        nb_haul = pm.NegativeBinomial.dist(mu=mu_haul, alpha=alpha_haul)
        
        # The observed total_points are drawn from this mixture
        total_points_likelihood = pm.Mixture(
            'total_points_likelihood', 
            w=pm.math.stack([1.0 - w, w]).T, 
            comp_dists=[nb_blank, nb_haul],
            observed=y.values
        )
        
    print("Model definition complete.")
    return bimodal_model

def plot_posterior_predictive(idata, fpl_model, y, player_name):
    """
    Generates and plots the posterior predictive distribution against the
    actual historical data histogram.
    """
    print("\n--- Generating Posterior Predictive Plot ---")

    # Generate posterior predictive samples
    with fpl_model:
        posterior_predictive = pm.sample_posterior_predictive(idata)

    # Extract the simulated scores
    simulated_scores = posterior_predictive.posterior_predictive['total_points_likelihood']

    # Set the plot limit to the 99.9th percentile of the simulated data
    plot_limit = int(np.percentile(simulated_scores, 99.9))
    max_score = max(y.max(), plot_limit, 25)
    score_range = np.arange(max_score + 1)

    # Reshape the data
    n_chains = simulated_scores.sizes['chain']
    n_draws = simulated_scores.sizes['draw']
    simulated_scores_reshaped = simulated_scores.values.reshape(n_chains * n_draws, -1)

    # --- THIS IS THE FIX ---
    # We clip the simulated scores to max_score. This prevents bincount from creating
    # arrays of different lengths when it encounters an outlier.
    avg_pred_pmf = np.mean([
        np.bincount(np.clip(game_samples, 0, max_score), minlength=len(score_range)) / len(game_samples)
        for game_samples in simulated_scores_reshaped.T
    ], axis=0)
    # --- END FIX ---

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(y, bins=np.arange(y.max() + 2) - 0.5, density=True,
            color='skyblue', edgecolor='black', alpha=0.7, label='Historical Actual Scores')

    ax.plot(score_range, avg_pred_pmf, 'o-', color='red',
            label='Model Posterior Predictive')
    
    ax.set_xlim(-1, max_score + 1)

    ax.set_title(f'Posterior Predictive Check for {player_name}', fontsize=16)
    ax.set_xlabel('Total Points', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(f"{player_name}_posterior_predictive.png")
    print(f"Posterior predictive plot saved to {player_name}_posterior_predictive.png")

if __name__ == "__main__":
    # Define paths and player to model
    DATA_PATH = Path("../data/processed/master_feature_dataset.csv")
    PLAYER_TO_MODEL = "Erling Haaland"

    try:
        # Load the master dataset
        master_df = pd.read_csv(DATA_PATH)

        # --- Step 2 ---
        X_player, y_player, feature_names = prepare_player_data(master_df, PLAYER_TO_MODEL)

        # --- Step 3 ---
        fpl_model = define_bimodal_model(X_player, y_player, feature_names)

        # --- Step 4: Run the MCMC Inference ---
        # This is the step that trains the model. It can take a few minutes.
        print("\n--- Step 4: Running MCMC Inference ---")
        with fpl_model:
            # The 'idata' object will contain all the results
            idata = pm.sample(2000, tune=1000, cores=8)
        
        print("\nInference complete.")

        # --- Step 5: Analyze and Visualize Results ---
        print("\n--- Step 5: Analyzing Results ---")
        # Print a summary of the posterior distributions for key parameters
        summary = az.summary(idata, var_names=['beta_w_intercept', 'beta_blank_intercept', 'beta_haul_intercept', 'alpha_blank', 'alpha_haul'])
        print(summary)
        
        # NEW: Plot the posterior predictive check
        plot_posterior_predictive(idata, fpl_model, y_player, PLAYER_TO_MODEL)

    except FileNotFoundError:
        print(f"Error: Master dataset not found at {DATA_PATH}")
    except ValueError as e:
        print(e)
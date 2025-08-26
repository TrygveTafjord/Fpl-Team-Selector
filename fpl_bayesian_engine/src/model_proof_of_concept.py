import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import pytensor.tensor as pt 
import matplotlib.pyplot as plt
import pytensor
from pathlib import Path
from scipy.stats import nbinom
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
    """
    Defines a more robust Bayesian bimodal model with stronger, more informative priors
    to better replicate the distinct peaks seen in frequentist fits.
    """
    print(f"\n--- Step 3: Defining the Bayesian Bimodal Model with Strong Priors ---")
    
    with pm.Model() as bimodal_model:
        # --- Priors for the regression coefficients (betas) ---
        
        #  Priors for the "Haul" probability (w) regression
        # A mean of -1.5 on the logit scale corresponds to a probability of sigmoid(-1.5) ~= 18%.
        # This reflects that hauls are less common than blanks.
        beta_w_intercept = pm.Normal('beta_w_intercept', mu=-1.5, sigma=0.5)
        beta_w_coeffs = pm.Normal('beta_w_coeffs', mu=0.0, sigma=0.5, shape=len(features))

        # 2. Priors for the "Blank" component (mu1) regression
        # Centered on a mean of ~2 points (log(2) ~= 0.7)
        beta_blank_intercept = pm.Normal('beta_blank_intercept', mu=np.log(2.5), sigma=0.25) 
        beta_blank_coeffs = pm.Normal('beta_blank_coeffs', mu=0.0, sigma=0.5, shape=len(features))

        # 3. Priors for the "Haul" component (mu2) OFFSET regression
        # A haul adds about 7-8 points to a blank. We'll center the log-offset there.
        # We use a very tight sigma (0.2) to strongly discourage the offset from becoming small.
        beta_haul_offset_intercept = pm.Normal('beta_haul_intercept', mu=np.log(7.5), sigma=0.2)
        beta_haul_offset_coeffs = pm.Normal('beta_offset_coeffs', mu=0.0, sigma=0.5, shape=len(features))

        # Priors for the dispersion parameters (alpha)
        # HalfNormal priors prevent alphas from becoming too large, keeping the peaks defined.
        alpha_blank = pm.HalfNormal('alpha_blank', sigma=0.5)
        alpha_haul = pm.HalfNormal('alpha_haul', sigma=1.0) # Hauls can have more variance

        # Link Functions (Regression Equations) 
        
        # Equation for the probability of a haul (w)
        w_logit = beta_w_intercept + pm.math.dot(X.values, beta_w_coeffs)
        w = pm.Deterministic('w', pm.math.sigmoid(w_logit))
        
        # Equation for the mean of the "Blank" component (mu1)
        mu_blank_log = beta_blank_intercept + pm.math.dot(X.values, beta_blank_coeffs)
        mu_blank = pm.Deterministic('mu_blank', pm.math.exp(mu_blank_log))
        
        # Equation for the mean of the "Haul" component (mu2)
        mu_haul_offset_log = beta_haul_offset_intercept + pm.math.dot(X.values, beta_haul_offset_coeffs)
        mu_haul = pm.Deterministic('mu_haul', mu_blank + pm.math.exp(mu_haul_offset_log))

        # Likelihood Function 
        
        nb_blank = pm.Poisson.dist(mu=mu_blank, alpha=alpha_blank)
        nb_haul = pm.Poisson.dist(mu=mu_haul, alpha=alpha_haul)
        
        # The mixture weight matrix needs to have shape (n_observations, n_components)
        weights = pt.stack([1.0 - w, w], axis=1)

        # The observed total_points are drawn from this mixture
        total_points_likelihood = pm.Mixture(
            'total_points_likelihood', 
            w=weights, 
            comp_dists=[nb_blank, nb_haul],
            observed=y.values
        )
        
    print("Model definition complete.")
    return bimodal_model

def create_frequentist_style_plot(player_name, idata, y_player, X_player_scaled):
    """
    Creates and saves a plot for the Bayesian model that mimics the style of the frequentist plots.
    This function is called right after model inference is complete.
    """
    print("\n--- Generating Frequentist-Style Plot ---")
    
    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(8, 7))

    # 1. Plot the original histogram (using frequency, not density)
    total_games = len(y_player)
    # Using seaborn for a slightly cleaner look, but ax.hist would also work
    import seaborn as sns
    sns.histplot(y_player, ax=ax, bins=np.arange(0, y_player.max() + 2) - 0.5, 
                 color='skyblue', stat='count', label='Observed Data')

    # 2. Extract the mean posterior parameters for an "average" match
    posterior = idata.posterior
    avg_features = X_player_scaled.mean(axis=0).values

    w_coeffs_dot = np.einsum('k,cdk->cd', avg_features, posterior['beta_w_coeffs'])
    w_logit_samples = posterior['beta_w_intercept'] + w_coeffs_dot
    avg_w = (1 / (1 + np.exp(-w_logit_samples))).mean().item()

    blank_coeffs_dot = np.einsum('k,cdk->cd', avg_features, posterior['beta_blank_coeffs'])
    mu_blank_log_samples = posterior['beta_blank_intercept'] + blank_coeffs_dot
    avg_mu_blank = np.exp(mu_blank_log_samples).mean().item()
    
    haul_offset_coeffs_dot = np.einsum('k,cdk->cd', avg_features, posterior['beta_offset_coeffs'])
    mu_haul_offset_log_samples = posterior['beta_haul_intercept'] + haul_offset_coeffs_dot
    avg_mu_haul = (np.exp(mu_blank_log_samples) + np.exp(mu_haul_offset_log_samples)).mean().item()

    avg_alpha_blank = posterior['alpha_blank'].mean().item()
    avg_alpha_haul = posterior['alpha_haul'].mean().item()

    # 3. Plot the fitted distribution and its components
    x_plot = np.arange(0, y_player.max() + 5)
    
    p_blank = avg_alpha_blank / (avg_alpha_blank + avg_mu_blank)
    comp1_pmf = (1 - avg_w) * nbinom.pmf(x_plot, n=avg_alpha_blank, p=p_blank)
    ax.plot(x_plot, comp1_pmf * total_games, color='darkorange', linestyle='--', label=f'Blank (μ={avg_mu_blank:.1f})')
    
    p_haul = avg_alpha_haul / (avg_alpha_haul + avg_mu_haul)
    comp2_pmf = avg_w * nbinom.pmf(x_plot, n=avg_alpha_haul, p=p_haul)
    ax.plot(x_plot, comp2_pmf * total_games, color='purple', linestyle='--', label=f'Haul (μ={avg_mu_haul:.1f})')

    fitted_pmf = comp1_pmf + comp2_pmf
    ax.plot(x_plot, fitted_pmf * total_games, color='red', linewidth=2.5, label='Fitted Bimodal Dist.')

    # 4. Final plot formatting
    avg_score = y_player.mean()
    ax.set_title(f"{player_name}\n(Avg: {avg_score:.2f} over {total_games} games)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(np.arange(0, y_player.max() + 2, step=max(1, ((y_player.max() + 1) // 5))))
    ax.legend(fontsize='small')
    ax.set_xlabel("Points in a Gameweek")
    ax.set_ylabel("Frequency (Counts)")
    
    # Save the figure
    output_filename = f"{player_name}_bayesian_fit_plot.png"
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.close() # Close the plot to free up memory


if __name__ == "__main__":
    # Define paths and player to model
    DATA_PATH = Path("../data/processed/master_feature_dataset.csv")
    PLAYER_TO_MODEL = "Bukayo Saka"

    try:
        # Load the master dataset
        master_df = pd.read_csv(DATA_PATH)
 
        X_player, y_player, feature_names = prepare_player_data(master_df, PLAYER_TO_MODEL)
 
        fpl_model = define_bimodal_model(X_player, y_player, feature_names)

        print("\n--- Step 4: Running MCMC Inference ---")
        with fpl_model:
            # The 'idata' object will contain all the results
            idata = pm.sample(2000, tune=1000, cores=8)
        
        print("\nInference complete.")

        print("\n--- Step 5: Analyzing Results ---")
        # Print a summary of the posterior distributions for key parameters
        summary = az.summary(idata, var_names=['beta_w_intercept', 'beta_blank_intercept', 'beta_haul_intercept', 'alpha_blank', 'alpha_haul'])
        print(summary)
        
        #Plot the posterior predictive check
        create_frequentist_style_plot(PLAYER_TO_MODEL, idata, y_player, X_player)

    except FileNotFoundError:
        print(f"Error: Master dataset not found at {DATA_PATH}")
    except ValueError as e:
        print(e)
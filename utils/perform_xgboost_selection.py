import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

def perform_xgboost_selection(df: pd.DataFrame, target_col: pd.DataFrame) -> list[str]:

    print("Starting XGBoost Feature Selection")
    
    X = df
    y = target_col

    # Ensure all data is numeric for XGBoost
    X = X.select_dtypes(include=np.number)

    # Train an Initial XGBoost Model to Get Importances 
    # This model is trained on all features to get a baseline importance ranking.
    print("Training initial model to get feature importances")
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=800, 
        max_depth=4,
        colsample_bytree=0.8,
        learning_rate=0.105,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1 # Use all available CPU cores
    )
    base_model.fit(X, y)

    # Find the Optimal Number of Features 
    # We will iterate through different feature importance thresholds.
    # For each threshold, we select a subset of features and evaluate a new model
    # using cross-validation to see how well it performs.
    thresholds = np.sort(base_model.feature_importances_)
    best_score = -np.inf
    best_feature_count = 0

    # Use TimeSeriesSplit for robust, time-aware cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    print(f"Testing {len(thresholds)} different feature thresholds...")
    for thresh in thresholds:
        # Select features using the current threshold
        selection = SelectFromModel(base_model, threshold=thresh, prefit=True)
        
        # Get the subset of features
        select_X = selection.transform(X)
        
        # If no features are selected, skip to the next threshold
        if select_X.shape[1] == 0:
            continue

        # Train a new model on the selected features using cross-validation
        # This gives a robust estimate of the performance for this feature subset.
        selection_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eta=0.1,
            max_depth=6,
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        )
        
        scores = cross_val_score(selection_model, select_X, y, cv=tscv, scoring='neg_mean_squared_error')
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_feature_count = select_X.shape[1]

        print(f"Thresh={thresh:.4f}, n={select_X.shape[1]}, MSE={-mean_score:.4f}, Best Score={-best_score:.4f}")

    print(f"\nOptimal number of features found: {best_feature_count} (with MSE: {-best_score:.4f})")

    # Get the Final List of Selected Features 
    # Now we use the best threshold to get the final list of feature names.
    final_selector = SelectFromModel(base_model, threshold=thresholds[len(thresholds) - best_feature_count], prefit=True)
    selected_feature_names = X.columns[(final_selector.get_support())].tolist()

    importances = dict(zip(X.columns, base_model.feature_importances_))
                       
    # Filter the dictionary to only include selected features and then sort
    selected_importances = {f: importances[f] for f in selected_feature_names}
    sorted_features = sorted(selected_importances, key=selected_importances.get, reverse=True)
    
    print("\nSelected features sorted by importance:")
    
    for f in sorted_features:
        print(f"{f}: {selected_importances[f]:.4f}")

    return sorted_features # Return the sorted list instead
    

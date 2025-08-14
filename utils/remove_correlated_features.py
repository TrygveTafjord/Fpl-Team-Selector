import pandas as pd
import numpy as np

def remove_correlated_features(df: pd.DataFrame, 
                            target_col: str = 'total_points',  # The target column for correlation checks
                              variance_threshold: float = 0.01,     # Minimum variance for a feature to be kept
                              corr_feature_threshold: float = 0.8,  # Maximum correlation allowed between two features
                              print_info: bool = True
                              ) -> list[str]:
    
    NUM_INITIAL_FEATURES = len(df.columns)
    # Remove low-variance features 
    feature_variance = df.var()
    low_variance_features = feature_variance[feature_variance < variance_threshold].index
    df = df.drop(columns=low_variance_features)
    if print_info:
        print(f"Removed {len(low_variance_features)} low-variance features.")

    # Remove highly correlated features

    # Calculate the correlation matrix
    correlation_matrix = df.corr()
    highly_correlated_features = list()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            # If the correlation is above the threshold, store the feature pair
            if abs(correlation_matrix.iloc[i, j]) > corr_feature_threshold:
                correlated_feature = [correlation_matrix.columns[i] , correlation_matrix.columns[j]]
                highly_correlated_features.append(correlated_feature)

    if print_info:
        print(f"Found {len(highly_correlated_features)} pairs of highly correlated features.")

    # Remove one feature from each pair based on correlation with target
    removed_features = set()
    num_removed = 0
    while len(highly_correlated_features) > 0:
        feature_pair = highly_correlated_features.pop(0)
        feature1, feature2 = feature_pair

        # Skip if either feature has already been removed
        if feature1 in removed_features or feature2 in removed_features:
            continue

        # Compare absolute correlation with target
        corr1 = abs(df.corr()['total_points'].get(feature1, 0))
        corr2 = abs(df.corr()['total_points'].get(feature2, 0))

        if corr1 < corr2:
            to_remove = feature1
            to_keep = feature2
        else:
            to_remove = feature2
            to_keep = feature1

        num_removed += 1
        removed_features.add(to_remove)
        df = df.drop(columns=[to_remove])
    
    if print_info:
        print(f"Removed {num_removed} features due to high correlation.")
    
    if print_info:
        print(f"Original number of features; {NUM_INITIAL_FEATURES}. Final number of features: {len(df.columns)}")
    
    return df
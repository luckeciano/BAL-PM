from scipy.stats import entropy
import numpy as np
import pandas as pd

def compute_calibration(df, calibration_dict, args):
    previous_alpha = 0.5
    for alpha in np.linspace(0.55, 0.95, args.num_bins):
        right_points = df[(df['First'] <= alpha) & (df['First'] > previous_alpha)]
        wrong_points = df[(df['Second'] <= alpha) & (df['Second'] > previous_alpha)]
        total_points = (len(right_points) + len(wrong_points))
        if total_points == 0:
            accuracy = 1.0
        else:
            accuracy = len(right_points) / (len(right_points) + len(wrong_points))
        previous_alpha = alpha
        calibration_dict[alpha].append(accuracy)    
    return calibration_dict


def compute_entropy(df):
    return df.apply(lambda row: entropy(row), axis=1)

def compute_uncertanties(dfs):
    # Compute single model entropies
    for df in dfs:
        df['entropy'] = compute_entropy(df[['First', 'Second']])
    
    
    for i, df in enumerate(dfs):
        # Add unique suffixes to the column names
        df.columns = [f"{col}_{i}" if col != 'id' else col for col in df.columns]

    # Use reduce to merge all dataframes
    from functools import reduce
    final_df = reduce(lambda left,right: pd.merge(left,right,on='id'), dfs)

    first_cols = [col for col in final_df.columns if 'First_' in col]
    second_cols = [col for col in final_df.columns if 'Second_' in col]
    entropy_cols = [col for col in final_df.columns if 'entropy_' in col]

    avg_first = final_df[first_cols].mean(axis=1)
    avg_second = final_df[second_cols].mean(axis=1)
    avg_entropy = final_df[entropy_cols].mean(axis=1)
    var_first = final_df[first_cols].var(axis=1)
    avg_df = pd.concat([avg_first, avg_second, avg_entropy, var_first], axis=1)
    avg_df.columns = ['First', 'Second', 'Aleatoric Uncertainty', 'Variance']

    
    avg_df['Predictive Uncertainty'] = compute_entropy(avg_df[['First', 'Second']])
    avg_df['Epistemic Uncertainty'] = avg_df['Predictive Uncertainty'] - avg_df['Aleatoric Uncertainty']
    return avg_df['Epistemic Uncertainty'], avg_df['Predictive Uncertainty'], avg_df['Aleatoric Uncertainty'], avg_df[['First', 'Second']], avg_df['Variance']
from scipy.stats import entropy
import pandas as pd
import numpy as np

def compute_ensemble_accuracy(ens_probs):
    ens_predictions = np.argmax(ens_probs, axis=1)
    accuracy = np.array(ens_predictions == 0, dtype=float).mean()
    return accuracy

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
    avg_df = pd.concat([avg_first, avg_second, avg_entropy, var_first, final_df['id']], axis=1)
    avg_df.columns = ['First', 'Second', 'Aleatoric Uncertainty', 'Variance', 'id']

    
    avg_df['Predictive Uncertainty'] = compute_entropy(avg_df[['First', 'Second']])
    avg_df['Epistemic Uncertainty'] = avg_df['Predictive Uncertainty'] - avg_df['Aleatoric Uncertainty']
    return avg_df['Epistemic Uncertainty'], avg_df['Predictive Uncertainty'], avg_df['Aleatoric Uncertainty'], avg_df[['First', 'Second']], avg_df['Variance'], avg_df['id']
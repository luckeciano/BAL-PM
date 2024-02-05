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

    #avg_df['EPIG'] = compute_epig(avg_df, dfs)
    
    avg_df['Predictive Uncertainty'] = compute_entropy(avg_df[['First', 'Second']])
    avg_df['Epistemic Uncertainty'] = avg_df['Predictive Uncertainty'] - avg_df['Aleatoric Uncertainty']
    return avg_df['Epistemic Uncertainty'], avg_df['Predictive Uncertainty'], avg_df['Aleatoric Uncertainty'], avg_df[['First', 'Second']], avg_df['Variance'], avg_df['id']

def compute_epig(avg_df, dfs):
    epig_df = pd.concat([avg_df] + dfs, axis=1)
    return epig_df.apply(lambda x: epig(x, epig_df, len(dfs)), axis=1)

def epig(x, epig_df, ens_size):
    M = 1000
    x_star = epig_df.sample(n=M, axis=0)
    epig_sample = x_star.apply(lambda row: compute_epig_sample(row, x, ens_size), axis=1)
    return epig_sample.mean()

def compute_epig_sample(x_star, x, ens_size):

    # Sample y_star
    x_star['y'] = np.random.choice(['First', 'Second'], p=[x_star['First'], 1- x_star['First']])

    # Sample y
    x['y'] =  np.random.choice(['First', 'Second'], p=[x['First'], 1 - x['First']])

    avg_p_x_star = x_star[x_star['y']]
    avg_p_x = x[x['y']]
    products = None
    for i in range(ens_size):
        p_x_star = x_star[f"{x_star['y']}_{i}"]
        p_x = x[f"{x['y']}_{i}"]
        if products is None:
            products = p_x * p_x_star
        else:
            products = products + p_x * p_x_star
    
    final_products = products / ens_size

    epig_sample = np.log(final_products) - np.log(avg_p_x) - np.log(avg_p_x_star)
    return epig_sample
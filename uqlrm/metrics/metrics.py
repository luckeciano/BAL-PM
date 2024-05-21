from scipy.stats import entropy
import pandas as pd
import numpy as np
import gc

def compute_accuracy(ens_probs):
    ens_predictions = np.argmax(ens_probs, axis=1)
    accuracy = np.array(ens_predictions == 0, dtype=float).mean()
    return accuracy

def compute_entropy(arr):
    return np.apply_along_axis(entropy, 1, arr)   

def compute_uncertanties(predictions, ids):
    # Convert ids to a numpy array
    ids_np = ids.values.flatten()

    # Compute single model entropies
    for i, pred in enumerate(predictions):
        pred = np.column_stack((pred, compute_entropy(pred[:, :2])))
        predictions[i] = pred
    
    # Concatenate arrays
    final_df = np.concatenate(predictions, axis=1)

    # Compute averages and variances
    avg_first = np.mean(final_df[:, ::3], axis=1)
    avg_second = np.mean(final_df[:, 1::3], axis=1)
    avg_entropy = np.mean(final_df[:, 2::3], axis=1)
    var_first = np.var(final_df[:, ::3], axis=1)
    
    # Compute predictive and epistemic uncertainties
    predictive_uncertainty = compute_entropy(np.column_stack((avg_first, avg_second)))
    epistemic_uncertainty = predictive_uncertainty - avg_entropy

    avg_df = pd.DataFrame({
        'First': avg_first,
        'Second': avg_second,
        'Aleatoric Uncertainty': avg_entropy,
        'Variance': var_first,
        'id': ids_np,
        'Predictive Uncertainty': predictive_uncertainty,
        'Epistemic Uncertainty': epistemic_uncertainty
    })

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
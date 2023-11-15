import pandas as pd
import argparse
from datasets import load_dataset
import os
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_entropy(df):
    return df.apply(lambda row: entropy(row), axis=1)

def compute_jsd(dfs):
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
    avg_df = pd.concat([avg_first, avg_second, avg_entropy], axis=1)
    avg_df.columns = ['First', 'Second', 'AvgEntropy']

    
    avg_df['EnsEntropy'] = compute_entropy(avg_df[['First', 'Second']])
    uq = avg_df['EnsEntropy'] - avg_df['AvgEntropy']
    return uq


def main(args):
    ckpts = [1] + list(range(args.min_ckpt, args.max_ckpt, args.steps_ckpt))
    for i in ckpts:      
        for mode in ["train", "eval", "test", "ood"]:
            ensemble_df = []
            for j in range(args.ensemble_size):
                    name = f"{args.experiment_name}_{j}"
                    datafile = os.path.join(args.experiment_prefix, name, name, f"checkpoint-{i}", f"eval_{mode}", "predictions.csv")
                    df = load_dataset("luckeciano/uqlrm_predictions", data_files=datafile)['train'].to_pandas()
                    ensemble_df.append(df)
        
            uq = compute_jsd(ensemble_df)
            sns.set_theme()
            sns.set_context("paper")
            sns.histplot(np.log(uq), stat='probability', label=mode, bins=100)
            plt.legend()
            ax = plt.gca()
            ax.set_xlim([-14.0, 0.0])
            ax.set_ylim([0, 0.06])
        plt.title(f"Epistemic Uncertainty Estimation - Checkpoint {i}")
        plt.savefig(f'./images/{args.experiment_name}/log_{args.experiment_name}_ckpt_{i}.eps', format='eps')
        plt.savefig(f'./images/{args.experiment_name}/log_{args.experiment_name}_ckpt_{i}.jpg')
        plt.savefig(f'./images/{args.experiment_name}/log_{args.experiment_name}_ckpt_{i}.svg')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate uncertainty plot.')
    parser.add_argument('experiment_prefix', type=str, default='scratch/lucelo/sft/results/', help='Name of the experiment')
    parser.add_argument('ensemble_size', type=int, help='Number of ensemble components')
    parser.add_argument('min_ckpt', type=int, default=1, help='Number of ensemble components')
    parser.add_argument('max_ckpt', type=int, help='Number of ensemble components')
    parser.add_argument('steps_ckpt', type=int, help='Number of ensemble components')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('uncertainty_measure', type=str, help='Uncertainty measure')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint (default: None)')
    args = parser.parse_args()
    main(args)
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
    avg_df = pd.concat([avg_first, avg_second, avg_entropy], axis=1)
    avg_df.columns = ['First', 'Second', 'AvgEntropy']

    
    avg_df['EnsEntropy'] = compute_entropy(avg_df[['First', 'Second']])
    uq = avg_df['EnsEntropy'] - avg_df['AvgEntropy']
    return uq, avg_df['EnsEntropy'], avg_df['AvgEntropy']


def plot_histogram(df, ax, label, bins, xlim, ylim, title):
    sns.histplot(df, stat='probability', label=label, bins=bins, ax=ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)

def save_plots(filepath, fig):
    fig.savefig(f'{filepath}.eps', format='eps')
    fig.savefig(f'{filepath}.jpg')
    fig.savefig(f'{filepath}.svg')

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def main(args):
    sns.set_theme()
    sns.set_context("paper")
    ckpts = [1] + list(range(args.min_ckpt, args.max_ckpt, args.steps_ckpt))
    create_directory(f'./images/{args.experiment_name}')
    create_directory(f'./images/{args.experiment_name}/epistemic')
    create_directory(f'./images/{args.experiment_name}/predictive')
    create_directory(f'./images/{args.experiment_name}/aleatoric')
    for i in ckpts:
        uncertainty_plots = {
            'LogEpistemic': plt.subplots(),
            'Epistemic': plt.subplots(),
            'Predictive': plt.subplots(),
            'Aleatoric': plt.subplots()
        }      
        for mode in ["train", "eval", "test", "ood"]:
            ensemble_df = []
            for j in range(args.ensemble_size):
                    name = f"{args.experiment_name}_{j}"
                    datafile = os.path.join(args.experiment_prefix, name, name, f"checkpoint-{i}", f"eval_{mode}", "predictions.csv")
                    try: 
                        df = load_dataset("luckeciano/uqlrm_predictions", data_files=datafile)['train'].to_pandas()
                        ensemble_df.append(df)
                    except:
                        continue
        
            print(f"Number of ensemble predictions loaded: {len(ensemble_df)}")
            epistemic, predictive, aleatoric = compute_uncertanties(ensemble_df)
            
            plot_histogram(np.log(epistemic), uncertainty_plots['LogEpistemic'][1], mode, 100, \
                           [-14.0, 0.0], [0, 0.06], f"Epistemic Uncertainty Estimation - Checkpoint {i} - Log Scale")
            plot_histogram(epistemic, uncertainty_plots['Epistemic'][1], mode, 100, \
                           [0.0, 0.5], [0, 0.1], f"Epistemic Uncertainty Estimation - Checkpoint {i}")
            plot_histogram(predictive, uncertainty_plots['Predictive'][1], mode, 100, \
                           [0.40, 0.70], [0, 0.1], f"Predictive Uncertainty Estimation - Checkpoint {i}")
            plot_histogram(aleatoric, uncertainty_plots['Aleatoric'][1], mode, 100, \
                           [0.35, 0.70], [0, 0.1], f"Aleatoric Uncertainty Estimation - Checkpoint {i}")

        uncertainty_plots['LogEpistemic'][1].legend()
        uncertainty_plots['Epistemic'][1].legend()
        uncertainty_plots['Predictive'][1].legend()
        uncertainty_plots['Aleatoric'][1].legend()

        save_plots(f'./images/{args.experiment_name}/epistemic/log_{args.experiment_name}_ckpt_{i}', uncertainty_plots['LogEpistemic'][0])
        save_plots(f'./images/{args.experiment_name}/epistemic/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Epistemic'][0])
        save_plots(f'./images/{args.experiment_name}/predictive/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Predictive'][0])
        save_plots(f'./images/{args.experiment_name}/aleatoric/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Aleatoric'][0])

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
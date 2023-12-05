import pandas as pd
import argparse
from datasets import load_dataset
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plot_utils import compute_uncertanties


def compute_uncertainty_corr(uq_df, err_df, unc_key, mode, corr_dict):
    rho, _ = stats.spearmanr(uq_df, err_df)
    corr_dict[unc_key][mode].append(rho)

def plot_unc_calibration(corr_dict, unc_key, ckpts):
    sns.set_theme()
    sns.set_context("paper")
    plt.legend()
    for mode in ["train", "eval", "test", "ood", "shuffled"]:
        # ax = plt.gca()
        # ax.set_xlim(unc_interval)
        # ax.set_ylim([0.0, 1.0])
        df = pd.DataFrame({'Checkpoint': ckpts, 'Spearman Correlation': corr_dict[unc_key][mode]})
        sns.lineplot(x='Checkpoint', y='Spearman Correlation', data=df, label=mode)
        

    plt.title(f"{unc_key}-Error Correlation over Training")
    folder_name = unc_key.replace(" ", "")
    create_directory(f'./images/{args.experiment_name}/uncertainty_calibration/{folder_name}')
    plt.savefig(f'./images/{args.experiment_name}/uncertainty_calibration/{folder_name}/{args.experiment_name}.svg')
    plt.savefig(f'./images/{args.experiment_name}/uncertainty_calibration/{folder_name}/{args.experiment_name}.eps', format='eps')
    plt.savefig(f'./images/{args.experiment_name}/uncertainty_calibration/{folder_name}/{args.experiment_name}.jpg')
    plt.close()


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def main(args):
    ckpts = [1] + list(range(args.min_ckpt, args.max_ckpt, args.steps_ckpt))

    corr_dict = {}
    for unc in ['Epistemic Uncertainty', 'Predictive Uncertainty', 'Aleatoric Uncertainty']:
        corr_dict[unc] = {}
        for mode in ["train", "eval", "test", "ood", "shuffled"]:
            corr_dict[unc][mode] = []

    for i in ckpts:   
        create_directory(f'./images/{args.experiment_name}')
        create_directory(f'./images/{args.experiment_name}/uncertainty_calibration')    
        for mode in ["train", "eval", "test", "ood", "shuffled"]:
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
            epistemic, predictive, aleatoric, ens_predictions = compute_uncertanties(ensemble_df)
            model_error = ens_predictions['First'] < 0.5
            compute_uncertainty_corr(epistemic, model_error, "Epistemic Uncertainty", mode, corr_dict)
            compute_uncertainty_corr(predictive, model_error, "Predictive Uncertainty", mode, corr_dict)
            compute_uncertainty_corr(aleatoric, model_error, "Aleatoric Uncertainty", mode, corr_dict)
            

    plot_unc_calibration(corr_dict, "Epistemic Uncertainty", ckpts)
    plot_unc_calibration(corr_dict, "Predictive Uncertainty", ckpts)
    plot_unc_calibration(corr_dict, "Aleatoric Uncertainty", ckpts)
                

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate uncertainty plot.')
    parser.add_argument('experiment_prefix', type=str, default='scratch/lucelo/sft/results/', help='Name of the experiment')
    parser.add_argument('ensemble_size', type=int, help='Number of ensemble components')
    parser.add_argument('min_ckpt', type=int, default=1, help='Number of ensemble components')
    parser.add_argument('max_ckpt', type=int, help='Number of ensemble components')
    parser.add_argument('steps_ckpt', type=int, help='Number of ensemble components')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('uncertainty_measure', type=str, help='Uncertainty measure')
    args = parser.parse_args()
    main(args)
import pandas as pd
import argparse
from datasets import load_dataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from metrics import compute_uncertanties
from utils import softmax
import glob
import copy
from scipy import stats

def plot_histogram(df, ax, label, bins, xlim, ylim, title):
    sns.histplot(df, stat='probability', label=label, bins=bins, ax=ax, alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)

def plot_correlation(corr, ax, label, title):
    cleaned_corr = [0 if np.isnan(x) else x for x in corr]
    sns.lineplot(x=list(range(len(cleaned_corr))), y=cleaned_corr, ax=ax, label=label)
    ax.set_title(title)


def plot_density(df, ax, label, bins, xlim, ylim, title):
    sns.histplot(df, cumulative=True, stat='density', kde=False, label=label, bins=bins, ax=ax, alpha=0.5)
    sns.kdeplot(df, cumulative=True, bw_adjust=0.5, color='k', linestyle='--')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)

def plot_retention_curve(uq_df, pred_df, ax, label, title, unc_key):
    uq_pred_df = pd.concat([uq_df, pred_df], axis=1).sort_values(by=unc_key, ascending=True)
    

    errors = [0.0]
    fraction_rate = [0.0]

    total_rows = len(uq_pred_df)
    preds = [0] * total_rows
    idx = 0
    for _, row in uq_pred_df.iterrows():
        fraction = (idx + 1) / total_rows
        preds[idx] = 0 if row['First'] >= 0.5 else 1 
        err = sum(preds) / total_rows
        errors.append(err)
        fraction_rate.append(fraction)
        idx += 1
    
    sns.lineplot(x=fraction_rate, y=errors, label=label, ax=ax)
    ax.set_title(title)

def save_plots(filepath, fig, jpg_only=False):
    if not jpg_only:
        fig.savefig(f'{filepath}.eps', format='eps')
        fig.savefig(f'{filepath}.svg')
    fig.savefig(f'{filepath}.jpg')
    

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def compute_uncertainty_corr(uq_df, err_df, unc_key, mode, corr_dict):
    rho, _ = stats.spearmanr(uq_df, err_df)
    corr_dict[unc_key][mode].append(rho)

def main(args):
    sns.set_theme()
    sns.set_context("paper")
    ckpts = list(range(args.min_ckpt, args.max_ckpt, args.steps_ckpt))
    create_directory(f'./images/{args.experiment_name}')
    create_directory(f'./images/{args.experiment_name}/retention')
    create_directory(f'./images/{args.experiment_name}/epistemic')
    create_directory(f'./images/{args.experiment_name}/variance')
    create_directory(f'./images/{args.experiment_name}/predictive')
    create_directory(f'./images/{args.experiment_name}/aleatoric')
    create_directory(f'./images/{args.experiment_name}/softmax_epistemic')
    create_directory(f'./images/{args.experiment_name}/power_epistemic')
    create_directory(f'./images/{args.experiment_name}/retention/epistemic')
    create_directory(f'./images/{args.experiment_name}/retention/variance')
    create_directory(f'./images/{args.experiment_name}/retention/predictive')
    create_directory(f'./images/{args.experiment_name}/retention/aleatoric')
    create_directory(f'./images/{args.experiment_name}/correlation')

    for i in ckpts:
        uncertainty_plots = {
            'LogEpistemic': plt.subplots(),
            'Epistemic': plt.subplots(),
            'LogVariance': plt.subplots(),
            'Variance': plt.subplots(),
            'Predictive': plt.subplots(),
            'Aleatoric': plt.subplots(),
            'SoftEpistemic': plt.subplots(),
            'PowerEpistemic': plt.subplots(),
        }

        retention_plots = {
            'Epistemic': plt.subplots(),
            'Variance': plt.subplots(),
            'Predictive': plt.subplots(),
            'Aleatoric': plt.subplots(),
        }

        corr_plots = {
            'train': plt.subplots(),
            'test': plt.subplots(),
            'eval': plt.subplots(),
            'ood': plt.subplots(),
        }      
        for mode in ["train", "ood", "eval", "test"]:#,"eval", "shuffled"]:
            files = glob.glob(f"/users/lucelo/mini_vpo/results/vpo_many_equal_final*/predictions/*/*/eval_{mode}/predictions.csv")
            files.extend(glob.glob(f"/users/lucelo/mini_vpo/results/vpo_multi*/predictions/*/*/eval_{mode}/predictions.csv"))
            ensemble_df = []
            for file in files:
                df = pd.read_csv(file, header=0)
                ensemble_df.append(df)

            corrs_var = []
            corrs_ep = []
            corrs_pred = []
            corrs_alea = []
            for ens_size in range(1, 75, 1):
                print(f"Number of ensemble predictions loaded: {ens_size}")
                ensemble_candidates = copy.deepcopy(ensemble_df[:ens_size])
                epistemic, predictive, aleatoric, ens_predictions, var_predictions, _ = compute_uncertanties(ensemble_candidates)
            
                # plot_histogram(np.log(epistemic), uncertainty_plots['LogEpistemic'][1], mode, 100, \
                #             [-14.0, 0.0], [0, 0.06], f"Epistemic Uncertainty Estimation - Checkpoint {i} - Log Scale")
                # plot_histogram(epistemic, uncertainty_plots['Epistemic'][1], mode, 100, \
                #             [0.0, 0.5], [0, 0.1], f"Epistemic Uncertainty Estimation - Checkpoint {i}")
                # plot_histogram(var_predictions, uncertainty_plots['Variance'][1], mode, 100, \
                #                [0.0, 0.5], [0, 0.1], f"Epistemic Uncertainty Estimation - Checkpoint {i}")
                # plot_histogram(np.log(var_predictions), uncertainty_plots['LogVariance'][1], mode, 100, \
                #                [-14.0, 0.0], [0, 0.1], f"Epistemic Uncertainty Estimation - Checkpoint {i} - Log Scale")
                # plot_histogram(predictive, uncertainty_plots['Predictive'][1], mode, 100, \
                #                [0.40, 0.70], [0, 0.1], f"Predictive Uncertainty Estimation - Checkpoint {i}")
                # plot_histogram(aleatoric, uncertainty_plots['Aleatoric'][1], mode, 100, \
                            # [0.35, 0.70], [0, 0.1], f"Aleatoric Uncertainty Estimation - Checkpoint {i}")
                model_error = ens_predictions['First'] < 0.5
                rho_ep, _ = stats.spearmanr(epistemic, model_error)
                rho_pred, _ = stats.spearmanr(predictive, model_error)
                rho_var, _ = stats.spearmanr(var_predictions, model_error)
                rho_ale, _ = stats.spearmanr(aleatoric, model_error)
                print(f"Correlation Epistemic, Ens size: {ens_size}: {rho_ep}")
                print(f"Correlation Variance, Ens size: {ens_size}: {rho_var}")
                print(f"Correlation Predictive, Ens size: {ens_size}: {rho_pred}")
                corrs_ep.append(rho_ep)
                corrs_pred.append(rho_pred)
                corrs_var.append(rho_var)
                corrs_alea.append(rho_ale)


                plot_retention_curve(epistemic, ens_predictions, retention_plots['Epistemic'][1],  
                                    f'{mode}_{ens_size}', f"Retention Plot with Epistemic Uncertainty - Checkpoint {i}", "Epistemic Uncertainty")
                plot_retention_curve(var_predictions, ens_predictions, retention_plots['Variance'][1],  
                                    f'{mode}_{ens_size}', f"Retention Plot with Epistemic Uncertainty - Checkpoint {i}", "Variance")
                plot_retention_curve(predictive, ens_predictions, retention_plots['Predictive'][1],  
                                    f'{mode}_{ens_size}', f"Retention Plot with Predictive Uncertainty - Checkpoint {i}", "Predictive Uncertainty")
                plot_retention_curve(aleatoric, ens_predictions, retention_plots['Aleatoric'][1],  
                                    f'{mode}_{ens_size}', f"Retention Plot with Aleatoric Uncertainty - Checkpoint {i}", "Aleatoric Uncertainty")
            
            # soft_epistemic = softmax(epistemic, temperature=10.0)
            # power_epistemic = softmax(np.log(epistemic), temperature=10.0)

            # if mode is "train":
            #     plot_histogram(soft_epistemic, uncertainty_plots['SoftEpistemic'][1], mode, 100, \
            #                     [min(soft_epistemic), max(soft_epistemic)], [0, 0.5], f"Softmax Epistemic Uncertainty Estimation - Checkpoint {i}")
            #     plot_histogram(power_epistemic, uncertainty_plots['PowerEpistemic'][1], mode, 100, \
            #                     [min(power_epistemic), max(power_epistemic)], [0, 0.5], f"Power Epistemic Uncertainty Estimation - Checkpoint {i}")


            plot_correlation(corrs_ep, corr_plots[mode][1], 'Epistemic Uncertainty', f'Correlation - {mode}')
            plot_correlation(corrs_pred, corr_plots[mode][1], 'Predictive Uncertainty', f'Correlation - {mode}')
            plot_correlation(corrs_var, corr_plots[mode][1], 'Variance', f'Correlation - {mode}')
            plot_correlation(corrs_alea, corr_plots[mode][1], 'Aleatoric Uncertainty', f'Correlation - {mode}')

        uncertainty_plots['LogEpistemic'][1].legend()
        uncertainty_plots['Epistemic'][1].legend()
        uncertainty_plots['Variance'][1].legend()
        uncertainty_plots['LogVariance'][1].legend()
        uncertainty_plots['Predictive'][1].legend()
        uncertainty_plots['Aleatoric'][1].legend()
        uncertainty_plots['SoftEpistemic'][1].legend()
        uncertainty_plots['PowerEpistemic'][1].legend()


        retention_plots['Epistemic'][1].legend()
        retention_plots['Predictive'][1].legend()
        retention_plots['Aleatoric'][1].legend()

        corr_plots['train'][1].legend()
        corr_plots['test'][1].legend()
        corr_plots['eval'][1].legend()
        corr_plots['ood'][1].legend()

        save_plots(f'./images/{args.experiment_name}/epistemic/log_{args.experiment_name}_ckpt_{i}', uncertainty_plots['LogEpistemic'][0], jpg_only=True)
        save_plots(f'./images/{args.experiment_name}/epistemic/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Epistemic'][0], jpg_only=True)
        save_plots(f'./images/{args.experiment_name}/variance/log_{args.experiment_name}_ckpt_{i}', uncertainty_plots['LogVariance'][0], jpg_only=True)
        save_plots(f'./images/{args.experiment_name}/variance/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Variance'][0], jpg_only=True)
        save_plots(f'./images/{args.experiment_name}/predictive/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Predictive'][0], jpg_only=True)
        save_plots(f'./images/{args.experiment_name}/aleatoric/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Aleatoric'][0], jpg_only=True)
        save_plots(f'./images/{args.experiment_name}/softmax_epistemic/{args.experiment_name}_ckpt_{i}', uncertainty_plots['SoftEpistemic'][0], jpg_only=True)
        save_plots(f'./images/{args.experiment_name}/power_epistemic/{args.experiment_name}_ckpt_{i}', uncertainty_plots['PowerEpistemic'][0], jpg_only=True)

        save_plots(f'./images/{args.experiment_name}/retention/epistemic/{args.experiment_name}_ckpt_{i}', retention_plots['Epistemic'][0])
        save_plots(f'./images/{args.experiment_name}/retention/variance/{args.experiment_name}_ckpt_{i}', retention_plots['Variance'][0])
        save_plots(f'./images/{args.experiment_name}/retention/predictive/{args.experiment_name}_ckpt_{i}', retention_plots['Predictive'][0])
        save_plots(f'./images/{args.experiment_name}/retention/aleatoric/{args.experiment_name}_ckpt_{i}', retention_plots['Aleatoric'][0])

        save_plots(f'./images/{args.experiment_name}/correlation/{args.experiment_name}_train', corr_plots['train'][0])
        save_plots(f'./images/{args.experiment_name}/correlation/{args.experiment_name}_test', corr_plots['test'][0])
        save_plots(f'./images/{args.experiment_name}/correlation/{args.experiment_name}_eval', corr_plots['eval'][0])
        save_plots(f'./images/{args.experiment_name}/correlation/{args.experiment_name}_ood', corr_plots['ood'][0])

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
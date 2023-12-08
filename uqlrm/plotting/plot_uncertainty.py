import pandas as pd
import argparse
from datasets import load_dataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from metrics import compute_uncertanties

def plot_histogram(df, ax, label, bins, xlim, ylim, title):
    sns.histplot(df, stat='probability', label=label, bins=bins, ax=ax)
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
    create_directory(f'./images/{args.experiment_name}/retention')
    create_directory(f'./images/{args.experiment_name}/epistemic')
    create_directory(f'./images/{args.experiment_name}/variance')
    create_directory(f'./images/{args.experiment_name}/predictive')
    create_directory(f'./images/{args.experiment_name}/aleatoric')
    create_directory(f'./images/{args.experiment_name}/retention/epistemic')
    create_directory(f'./images/{args.experiment_name}/retention/variance')
    create_directory(f'./images/{args.experiment_name}/retention/predictive')
    create_directory(f'./images/{args.experiment_name}/retention/aleatoric')
    for i in ckpts:
        uncertainty_plots = {
            'LogEpistemic': plt.subplots(),
            'Epistemic': plt.subplots(),
            'LogVariance': plt.subplots(),
            'Variance': plt.subplots(),
            'Predictive': plt.subplots(),
            'Aleatoric': plt.subplots()
        }

        retention_plots = {
            'Epistemic': plt.subplots(),
            'Variance': plt.subplots(),
            'Predictive': plt.subplots(),
            'Aleatoric': plt.subplots()
        }      
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
            epistemic, predictive, aleatoric, ens_predictions, var_predictions = compute_uncertanties(ensemble_df)
            
            plot_histogram(np.log(epistemic), uncertainty_plots['LogEpistemic'][1], mode, 100, \
                           [-14.0, 0.0], [0, 0.06], f"Epistemic Uncertainty Estimation - Checkpoint {i} - Log Scale")
            plot_histogram(epistemic, uncertainty_plots['Epistemic'][1], mode, 100, \
                           [0.0, 0.5], [0, 0.1], f"Epistemic Uncertainty Estimation - Checkpoint {i}")
            plot_histogram(var_predictions, uncertainty_plots['Variance'][1], mode, 100, \
                           [0.0, 0.5], [0, 0.1], f"Epistemic Uncertainty Estimation - Checkpoint {i}")
            plot_histogram(np.log(var_predictions), uncertainty_plots['LogVariance'][1], mode, 100, \
                           [-14.0, 0.0], [0, 0.1], f"Epistemic Uncertainty Estimation - Checkpoint {i} - Log Scale")
            plot_histogram(predictive, uncertainty_plots['Predictive'][1], mode, 100, \
                           [0.40, 0.70], [0, 0.1], f"Predictive Uncertainty Estimation - Checkpoint {i}")
            plot_histogram(aleatoric, uncertainty_plots['Aleatoric'][1], mode, 100, \
                           [0.35, 0.70], [0, 0.1], f"Aleatoric Uncertainty Estimation - Checkpoint {i}")
            
            plot_retention_curve(epistemic, ens_predictions, retention_plots['Epistemic'][1],  
                                 mode, f"Retention Plot with Epistemic Uncertainty - Checkpoint {i}", "Epistemic Uncertainty")
            plot_retention_curve(var_predictions, ens_predictions, retention_plots['Variance'][1],  
                                 mode, f"Retention Plot with Epistemic Uncertainty - Checkpoint {i}", "Variance")
            plot_retention_curve(predictive, ens_predictions, retention_plots['Predictive'][1],  
                                 mode, f"Retention Plot with Predictive Uncertainty - Checkpoint {i}", "Predictive Uncertainty")
            plot_retention_curve(aleatoric, ens_predictions, retention_plots['Aleatoric'][1],  
                                 mode, f"Retention Plot with Aleatoric Uncertainty - Checkpoint {i}", "Aleatoric Uncertainty")

        uncertainty_plots['LogEpistemic'][1].legend()
        uncertainty_plots['Epistemic'][1].legend()
        uncertainty_plots['Variance'][1].legend()
        uncertainty_plots['LogVariance'][1].legend()
        uncertainty_plots['Predictive'][1].legend()
        uncertainty_plots['Aleatoric'][1].legend()

        retention_plots['Epistemic'][1].legend()
        retention_plots['Predictive'][1].legend()
        retention_plots['Aleatoric'][1].legend()

        save_plots(f'./images/{args.experiment_name}/epistemic/log_{args.experiment_name}_ckpt_{i}', uncertainty_plots['LogEpistemic'][0])
        save_plots(f'./images/{args.experiment_name}/epistemic/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Epistemic'][0])
        save_plots(f'./images/{args.experiment_name}/variance/log_{args.experiment_name}_ckpt_{i}', uncertainty_plots['LogVariance'][0])
        save_plots(f'./images/{args.experiment_name}/variance/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Variance'][0])
        save_plots(f'./images/{args.experiment_name}/predictive/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Predictive'][0])
        save_plots(f'./images/{args.experiment_name}/aleatoric/{args.experiment_name}_ckpt_{i}', uncertainty_plots['Aleatoric'][0])

        save_plots(f'./images/{args.experiment_name}/retention/epistemic/{args.experiment_name}_ckpt_{i}', retention_plots['Epistemic'][0])
        save_plots(f'./images/{args.experiment_name}/retention/variance/{args.experiment_name}_ckpt_{i}', retention_plots['Variance'][0])
        save_plots(f'./images/{args.experiment_name}/retention/predictive/{args.experiment_name}_ckpt_{i}', retention_plots['Predictive'][0])
        save_plots(f'./images/{args.experiment_name}/retention/aleatoric/{args.experiment_name}_ckpt_{i}', retention_plots['Aleatoric'][0])

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
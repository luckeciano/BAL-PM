import pandas as pd
import argparse
from datasets import load_dataset
import os
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def compute_calibration(df, calibration_dict):
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


def main(args):
    ckpts = [1] + list(range(args.min_ckpt, args.max_ckpt, args.steps_ckpt))
    for i in ckpts:      
        for mode in ["train", "eval", "test", "ood"]:
            calibration_dict = {}
            for alpha in np.linspace(0.55, 0.95, args.num_bins):
                calibration_dict[alpha] = []

            for j in range(args.ensemble_size):
                name = f"{args.experiment_name}_{j}"
                datafile = os.path.join(args.experiment_prefix, name, name, f"checkpoint-{i}", f"eval_{mode}", "predictions.csv")
                df = load_dataset("luckeciano/uqlrm_predictions", data_files=datafile)['train'].to_pandas()
                calibration_dict = compute_calibration(df, calibration_dict)

            # means, lb, up, alphas = compute_calibration_stats(calibration_dict)
            df = pd.DataFrame.from_dict(calibration_dict, orient='index').transpose()
            df_melt = df.melt(var_name='Confidence', value_name='Accuracy')
            sns.set_theme()
            sns.set_context("paper")
            ax = plt.gca()
            ax.set_xlim([0.5, 1.0])
            ax.set_ylim([0.5, 1.0])
            sns.lineplot(x='Confidence', y='Accuracy', data=df_melt, label=mode)
        plt.legend()

        plt.title(f"Model Calibration - Checkpoint {i}")
        x = np.linspace(0.55, 1.0, args.num_bins)
        sns.lineplot(x=x, y=x, color='black', linestyle='dashed', label="Ideal")
        plt.savefig(f'./images/{args.experiment_name}/model_calibration_{args.experiment_name}_ckpt_{i}.svg')
        plt.savefig(f'./images/model_calibration_{args.experiment_name}_ckpt_{i}.eps', format='eps')
        plt.savefig(f'./images/model_calibration_{args.experiment_name}_ckpt_{i}.jpg')
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
    parser.add_argument('num_bins', type=int, help='Uncertainty measure')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint (default: None)')
    args = parser.parse_args()
    main(args)
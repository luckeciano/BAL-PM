import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats
import seaborn as sns



def get_data(root_path, exp_name, metric, smoothing_window=3):
    folder_path = os.path.join(root_path, exp_name, metric)
    metric_col = f'{exp_name} - {metric_name}'

    # Get all csv files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize a list to store all data
    all_data = []

    # Iterate over all csv files
    for csv_file in csv_files:
        # Load the data
        data = pd.read_csv(os.path.join(folder_path, csv_file))
        # Append the data to all_data list
        all_data.append(data[[metric_col]])

    # Concatenate all data along the column axis (axis=1)
    concat_data = pd.concat(all_data, axis=1)
    smooth_data = concat_data.rolling(window=smoothing_window, min_periods=1).mean()
    return smooth_data

# Set the context to 'paper'
sns.set_context('paper', font_scale=2.0)


sns.set_style('darkgrid')
# Set the color palette to 'colorblind'
sns.set_palette('colorblind')


plt.figure(figsize=(8, 6))
# Create a color map
# cmap = plt.get_cmap('Pastel2')
root_path = '/users/lucelo/UQLRM/uqlrm/plotting/data'
metric = 'test_LogLikelihood'
exp_names = ['batchstateentropy_adp_rank_nounc_statenorm_reddit_7b', 'ep_adp_rank_bald_reddit_7b_final', 'baepm_reddit_7b_ratiolog_final'] #, 'al_batchent_nounc']
legend_dict = {
    'batchstateentropy_adp_rank_nounc_statenorm_reddit_7b': 'No Uncertainty Score',
    'ep_adp_rank_bald_reddit_7b_final': 'No Entropy Score',
    'baepm_reddit_7b_ratiolog_final': 'BAL-PM (ours)'
}
metric_name = f'train/ensemble/{metric}'


ax = plt.gca()

for i, exp_name in enumerate(exp_names):
    
    metric_col = f'{exp_name} - {metric_name}'
    concat_data = get_data(root_path, exp_name, metric)

    # Compute the mean and standard error for each row
    mean = concat_data[metric_col].mean(axis=1)
    stderr = concat_data[metric_col].apply(lambda x: stats.sem(x, axis=None, ddof=0), axis=1)
    confidence = 0.95
    ci = stderr * stats.t.ppf((1 + confidence) / 2., len(mean)-1)

    index = np.arange(len(mean)) + 1

    x = index * 320

    # Plot the data with the Pastel1 color map
    ax.plot(x, mean, label=legend_dict[exp_name])
    ax.fill_between(x, mean-stderr, mean+stderr, alpha=0.3)
    

    


# Add labels and title
ax.set_xlabel('Acquired Data')
ax.set_title('Log Likelihood ↑')
ax.set_xticks([2000, 8000, 12000, 16000, 20000, 24000])
ax.set_yticks([-0.62, -0.61, -0.63])
# ax.set_yticklabels([-0.62, -0.61])

ax.set_xlim(2000, 25000)
ax.set_ylim(-0.64, -0.6075)

ax.legend(loc='lower right')
plt.tight_layout()
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# Save the figure in EPS, PDF, and PNG formats
output_name = 'score_ablation_test_figure'
plt.savefig(f'{output_name}.eps', format='eps')
plt.savefig(f'{output_name}.pdf', format='pdf', dpi=1200)
plt.savefig(f'{output_name}.png', format='png')


# Show the plot
plt.show()
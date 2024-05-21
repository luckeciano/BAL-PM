import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats
import seaborn as sns



def get_data(root_path, exp_name, metric):
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
    return concat_data

# Set the context to 'paper'
sns.set_context('paper', font_scale=1.1)


sns.set_style('darkgrid')
# Set the color palette to 'colorblind'
sns.set_palette('colorblind')


plt.figure(figsize=(8, 6))
# Create a color map
# cmap = plt.get_cmap('Pastel2')
root_path = '/users/lucelo/UQLRM/uqlrm/plotting/data'
metric = 'eval_LogLikelihood'
exp_names = ['rand_adp_rank_bald_reddit_7b_final', 'ep_adp_powerbald8_bald_reddit_7b', 'ep_adp_powerbaldanneal_bald_reddit_7b', 'ep_adp_powerbald0.25_bald_reddit_7b'] #, 'al_batchent_nounc']
legend_dict = {
    'rand_adp_rank_bald_reddit_7b_final': 'Random Sampling',
    'ep_adp_rank_bald_reddit_7b_final': 'BALD',
    'ep_adp_powerbald8_bald_reddit_7b': 'PowerBald (\u03B2 = 8)',
    'ep_adp_powerbald0.25_bald_reddit_7b': 'PowerBald (\u03B2 = 0.25)',
    'ep_adp_powerbaldanneal_bald_reddit_7b': 'Annealed PowerBALD',
    'al_upperbound_hermes': 'Full Dataset'
}
metric_name = f'train/ensemble/{metric}'

full_data_exp_name = 'al_upperbound_hermes'
full_data_baseline = get_data(root_path, full_data_exp_name, metric)
mean = full_data_baseline[f'{full_data_exp_name} - {metric_name}'].mean(axis=1)


ax = plt.axes([0.06, 0.10, 0.4, 0.8])  # left, bottom, width, height
axins_inf = plt.axes([0.575, 0.05, 0.4, 0.44])  # left, bottom, width, height
axins_sup = plt.axes([0.575, 0.54, 0.4, 0.45])  # left, bottom, width, height
ax.axhline(mean.values[0], color='purple', linestyle='--', label=legend_dict[full_data_exp_name])
axins_inf.axhline(mean.values[0], color='purple', linestyle='--', label=legend_dict[full_data_exp_name])
axins_sup.axhline(mean.values[0],  color='purple', linestyle='--', label=legend_dict[full_data_exp_name])


for i, exp_name in enumerate(exp_names):
    
    metric_col = f'{exp_name} - {metric_name}'
    concat_data = get_data(root_path, exp_name, metric)

    # Compute the mean and standard error for each row
    mean = concat_data[metric_col].mean(axis=1)
    stderr = concat_data[metric_col].apply(lambda x: stats.sem(x, axis=None, ddof=0), axis=1)
    confidence = 0.95
    ci = stderr * stats.t.ppf((1 + confidence) / 2., len(mean)-1)

    index = np.arange(len(mean))

    x = index * 320

    # Plot the data with the Pastel1 color map
    ax.plot(x, mean, label=legend_dict[exp_name])
    ax.fill_between(x, mean-stderr, mean+stderr, alpha=0.3)

    
    
    # Plot the zoomed portion on the new axes
    axins_inf.plot(x, mean, label=legend_dict[exp_name])
    axins_inf.fill_between(x, mean-stderr, mean+stderr, alpha=0.3)
    axins_sup.plot(x, mean, label=legend_dict[exp_name])
    axins_sup.fill_between(x, mean-stderr, mean+stderr, alpha=0.3)

    
axins_inf.set_xlim(-100, 6000)
axins_inf.set_ylim(-0.682, -0.63)
axins_sup.set_xlim(8000, 24000)
axins_sup.set_ylim(-0.6325, -0.621)
axins_sup.set_xticks([8000, 12000, 16000, 20000, 24000])
ax.indicate_inset_zoom(axins_inf, edgecolor="black")
ax.indicate_inset_zoom(axins_sup, edgecolor="black")
# Add labels and title
ax.set_xlabel('Acquired Data')
ax.set_title('Log Likelihood ↑')
ax.set_xticks([0, 4000, 8000, 12000, 16000, 20000, 24000])

ax.set_xlim(-1000, 24100)
ax.set_ylim(-0.685, -0.615)

ax.legend(loc='lower right')
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# Save the figure in EPS, PDF, and PNG formats
output_name = 'test_plots_wandb'
plt.savefig(f'{output_name}.eps', format='eps')
plt.savefig(f'{output_name}.pdf', format='pdf', dpi=1200)
plt.savefig(f'{output_name}.png', format='png')


# Show the plot
plt.show()
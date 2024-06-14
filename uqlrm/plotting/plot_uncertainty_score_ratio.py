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

def plot_with_ci(data, ax, plot_label):
    # Compute the mean and standard error for each row
    mean = data[metric_col].mean(axis=1)
    stderr = data[metric_col].apply(lambda x: stats.sem(x, axis=None, ddof=0), axis=1)
    confidence = 0.95
    ci = stderr * stats.t.ppf((1 + confidence) / 2., len(mean)-1)

    index = np.arange(len(mean))

    x = index * 320

    # Plot the data with the Pastel1 color map
    ax.plot(x, mean, label=plot_label)
    ax.fill_between(x, mean-ci, mean+ci, alpha=0.3)
    

# Set the context to 'paper'
sns.set_context('paper', font_scale=1.5)


sns.set_style('darkgrid')
# Set the color palette to 'colorblind'
sns.set_palette('colorblind')


fig = plt.figure(figsize=(5, 5))
# fig_stack = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(111)
# ax_stack = fig.add_subplot(111)
# Create a color map
# cmap = plt.get_cmap('Pastel2')
root_path = '/users/lucelo/UQLRM/uqlrm/plotting/data/uncertainty_score_ratio'
metric = 'uncertainty_score_ratio_max'
exp_name = 'baepm_reddit_7b_ratiolog_final'
legend_dict = {
    'baepm_reddit_7b_ratiolog_final': '$\hat{U}(x, y_{1}, y_{2})$'
}
metric_name = f'train/batch_stats/{metric}'

    
metric_col = f'{exp_name} - {metric_name}'
concat_data = get_data(root_path, exp_name, '')

compl_data = 1.0 - concat_data

plot_with_ci(concat_data, ax, legend_dict[exp_name])
plot_with_ci(compl_data, ax, '$\hat{\mathcal{H}}({X_{tr} \cup \{x\}})$')
# plot_stacked(concat_data, compl_data, ax_stack, legend_dict[exp_name], 'Prompt Entropy')

# Add labels and title
ax.set_xlabel('Acquired Data')
ax.set_title('Score Ratio')
ax.set_xticks([0, 8000, 16000, 24000])

# ax.set_xlim(-1000, 24100)
# ax.set_ylim(-0.685, -0.615)

ax.legend(loc='lower right')
fig.tight_layout()
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# Save the figure in EPS, PDF, and PNG formats
output_name = 'uncertainty_score_ratio'
fig.savefig(f'{output_name}.eps', format='eps')
fig.savefig(f'{output_name}.pdf', format='pdf')
fig.savefig(f'{output_name}.png', format='png')

# fig_stack.savefig(f'{output_name}_stack.png', format='png')


# Show the plot
plt.show()
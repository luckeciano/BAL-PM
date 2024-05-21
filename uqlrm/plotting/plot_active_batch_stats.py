import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import glob

# Set the context to 'paper'
sns.set_context('paper', font_scale=2.0)

sns.set_style('darkgrid')

# Set the color palette to 'colorblind'
sns.set_palette('colorblind')

def get_csv_files(directory):
    os.chdir(directory)
    return glob.glob("*.csv")

prefix = '/users/lucelo/UQLRM/uqlrm/plotting/data/active_batch_visualization'
method2name = {
    os.path.join(prefix, 'random'): 'Random Sampling',
    os.path.join(prefix, 'bald'): 'BALD',
    # os.path.join(prefix, 'batch_ids_ep_adp_powerbaldanneal_bald_reddit_7b_idxs.csv'): 'AnnealedBALD',
    os.path.join(prefix, 'balpm'): 'BAL-PM (ours)'
}

def plot_with_stderr(exp, ax):
    mean = np.mean(exp, axis=0)
    sterr = np.std(exp, axis=0) / np.sqrt(len(exp))
    l, = ax.plot(x_axis, mean, label=v)
    ax.fill_between(x_axis, mean-sterr, mean+sterr, alpha=0.3)
    return l, v

fig = plt.figure(figsize=(15, 4))
ax0 = fig.add_subplot(1, 3, 1)
ax1 = fig.add_subplot(1, 3, 2)
ax2 = fig.add_subplot(1, 3, 3)
# cmap = plt.get_cmap('Dark2')
# Load the huggingface datasets
prompt_df = load_dataset("luckeciano/learning-to-summarize")['train'].to_pandas()
index = np.arange(75)
x_axis = index * 320


steps = [0]
plot_idx = 0
ls = []
labels = []
for k, v in method2name.items():
    csvs = get_csv_files(k)
    # Load the csv file
    generate_batch = True
    all_seeds_unique_prompts = []
    all_seeds_unique_total_ratios = []
    all_seeds_all_unique_total_ratios = []
    for csv in csvs:
        csv_data = pd.read_csv(csv, header=None)
        
        all_prompts = []
        all_unique_prompts = set()
        unique_total_ratios = []
        all_unique_total_ratios = []
        unique_prompts_num = []
        # Iterate over each row in the csv file
        for index, row in csv_data.iterrows():
            # Get the indices from the csv row
            indices = row.tolist()
            
            # Select the rows in X and Y with the same indices
            chosen_pts = prompt_df[prompt_df['id'].isin(indices)]

            chosen_prompts = sorted(chosen_pts['post'].values)

            if generate_batch:
                if index in steps:
                    with open(f'batch_prompts_{v}_{index}.txt', 'w') as f:
                        for item in chosen_prompts:
                            item = item.replace('\n', ' ')
                            f.write("%s\n" % item)
                    generate_batch = False

            unique_prompts = set(chosen_prompts)
            all_prompts.extend(chosen_prompts)
            all_unique_prompts = all_unique_prompts.union(unique_prompts)
            unique_total_ratios.append(len(unique_prompts) / len(chosen_prompts))
            all_unique_total_ratios.append(len(all_unique_prompts) / len(all_prompts))
            unique_prompts_num.append(len(all_unique_prompts))
            
        all_seeds_unique_prompts.append(unique_prompts_num)
        all_seeds_unique_total_ratios.append(unique_total_ratios)
        all_seeds_all_unique_total_ratios.append(all_unique_total_ratios)


    l, v = plot_with_stderr(all_seeds_unique_prompts, ax0)
    plot_with_stderr(all_seeds_unique_total_ratios, ax1)
    plot_with_stderr(all_seeds_all_unique_total_ratios, ax2)
    ls.append(l)
    labels.append(v)
        
        

    plot_idx += 1


ax0.set_title("Unique Acquired Prompts", fontsize=15)
ax1.set_title("Unique Prompts Ratio - Acquired Batch", fontsize=15)
ax2.set_title("Unique Prompts Ratio - Full Acquired Data", fontsize=15)


ax1.set_xlabel("Acquired Data", fontsize=15)
ax0.set_xlabel("Acquired Data", fontsize=15)
ax2.set_xlabel("Acquired Data", fontsize=15)

ax0.set_xticks([0, 8000, 16000, 24000])
ax0.set_yticks([0, 5000, 10000, 14000])
ax1.set_xticks([0, 8000, 16000, 24000])
ax1.set_yticks([0.5, 0.7, 0.9, 1.0])
ax2.set_xticks([0, 8000, 16000, 24000])
ax2.set_yticks([0.4, 0.6, 0.8, 1.0])

# handles, labels = ax2.get_legend_handles_labels()

# Create a unique legend box
fig.legend(handles=ls,     # The line objects
           labels=labels,   # The labels for each line
           loc="lower center",   # Position of legend
           borderaxespad=0.1,
            ncol=5, 
               bbox_to_anchor=(0.5, -0.05)   # Small spacing around legend box
           )
# fig.legend()
plt.tight_layout()
plt.savefig(f'batch_stats.png', format='png', bbox_inches='tight')
plt.savefig(f'batch_stats.pdf', format='pdf',bbox_inches='tight', dpi=1200)

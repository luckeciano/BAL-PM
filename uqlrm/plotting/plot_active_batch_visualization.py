import pandas as pd
from datasets import load_dataset
from sklearn.discriminant_analysis import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from umap import UMAP
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.stats import gaussian_kde as kde
from sklearn.neighbors import KernelDensity

def makeColours( vals ):
    colours = np.zeros( (len(vals),3) )
    norm = Normalize( vmin=vals.min(), vmax=vals.max() )

    #Can put any colormap you like here.
    colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

    return colours


# Load the csv file
csv_data = pd.read_csv('/users/lucelo/UQLRM/uqlrm/plotting/data/active_batch_visualization/batch_ids_batchstateentropy_adp_rank_nounc_statenorm_reddit_7b.csv', header=None)

# Load the huggingface datasets
prompt_df = load_dataset("luckeciano/learning-to-summarize")['train'].to_pandas()
rep_df = load_dataset("luckeciano/hermes-reddit-post-features")['train'].to_pandas()

# Initialize empty lists to store selected rows
selected_X = []
selected_Y = []



# Select the columns from 0 to 4095
feature_cols = [str(i) for i in range(4096)]
rep_features = rep_df[feature_cols]

# Create a StandardScaler object and fit to the training set
scaler = StandardScaler()
scaled_rep_fts = scaler.fit_transform(rep_features)

# Add Noise
scaled_rep_fts = scaled_rep_fts + np.random.normal(0, 1e-6, size=scaled_rep_fts.shape)

# Create a PCA object with 3 components and fit to the training set
# pca = PCA(n_components=4096)
# data_pca = pca.fit_transform(scaled_rep_fts)
# reductor = TSNE(n_components=3, random_state=42, verbose=2, n_iter=1000)
reductor = UMAP(n_components=2, random_state=42, verbose=2)
features_tsne = reductor.fit_transform(scaled_rep_fts)
df_tsne = pd.DataFrame(features_tsne, columns=['TSNE1', 'TSNE2'])
df_tsne['id'] = rep_df['id']

prompt_reps_df = pd.merge(prompt_df, df_tsne, on='id', how='inner')

fig = plt.figure(figsize=(20, 4))

steps = [0, 14, 24, 49, 74]

plot_idx = 0
# Iterate over each row in the csv file
for index, row in csv_data.iterrows():
    if index not in steps:
        continue
    # Get the indices from the csv row
    indices = row.tolist()

    plot_idx += 1
    
    # Select the rows in X and Y with the same indices
    chosen_pts = prompt_reps_df[prompt_reps_df['id'].isin(indices)]
    remaining_pts = prompt_reps_df[~prompt_reps_df['id'].isin(indices)]

    chosen_prompts = sorted(chosen_pts['post'].values)

    with open(f'batch_prompts_{index}.txt', 'w') as f:
        for item in chosen_prompts:
            item = item.replace('\n', ' ')
            f.write("%s\n" % item)

    # chosen_pts_rep = reductor.transform(pca.transform(scaler.transform(chosen_pts[feature_cols])))
    # remaining_pts_rep = reductor.transform(scaler.transform(remaining_pts[feature_cols]))

    # Add a subplot
    ax = fig.add_subplot(1, 5, plot_idx)

    # Create a 3D scatter plot of the three PCA components
    chosen_pts_rep = chosen_pts[['TSNE1', 'TSNE2']].values
    # densObj = kde( chosen_pts_rep.T )
    densObj = KernelDensity(kernel='gaussian').fit(chosen_pts_rep)
    colours = makeColours( densObj.score_samples( chosen_pts_rep ) )
    # ax.scatter(remaining_pts_rep[:, 0], remaining_pts_rep[:, 1], remaining_pts_rep[:, 2], color='gray', alpha=0.01, label='Remaining Points')
    ax.scatter(chosen_pts_rep[:, 0], chosen_pts_rep[:, 1], color='red', alpha=0.3, label='Chosen Points')
    ax.set_xlim([-20, 20])
    ax.set_ylim([-15, 25])
    # ax.set_zlim([-30, 40])
    ax.set_title(f'Step: {index}')


# plt.title('3D Scatter Plot of Test Set After PCA')
output_name = 'test_batch_visualization'
plt.savefig(f'{output_name}.eps', format='eps')
plt.savefig(f'{output_name}.pdf', format='pdf')
plt.savefig(f'{output_name}.png', format='png')

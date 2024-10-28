# Deep Bayesian Active Learning for Preference Modeling in Large Language Models

## Status
This repository contains the code and experiments for our paper **"Deep Bayesian Active Learning for Preference Modeling in Large Language Models"**. The codebase is currently stable and supports reproducing the main results presented in the paper.

## Description
This repository contains the code of the Bayesian Active Learner for Preference Modeling (BAL-PM). BAL-PM is a novel approach for actively selecting human preference triples to leverage human feedback more efficiently in the batch acquisiton setting. By targeting points with high epistemic uncertainty and maximizing entropy in the prompt distribution, BAL-PM avoids acquiring redundant samples and reduce preference labeling costs. Experiments show that BAL-PM can reduce the number of required preference labels by 33% to 68% across popular datasets, outperforming previous Bayesian acquisition methods in preference modeling.

## Getting Started

### Conda Environment

1. Create conda environment using:
    ```bash
    conda env create -f environment_cloud.yml
    ```

### Main Results
To reproduce the main results from the paper, we provide SLURM scripts for all experiments. If your setup does not involve SLURM, you can still use the bash commands in the script to run the experiments:

```bash
cd uqlrm/scripts/slurm
sbatch run_active_learning_adapters.sh
```

### Baselines

To reproduce baselines, please refer to the `scripts` directory and select the script associated to the baseline you want to reproduce.

### Reproducing Plots

To reproduce plots, please refer to the python scripts in the `plotting` directory. They generate plots based on the data in the `plotting/data` directory, which was manually collected from the W&B logs.

### Datasets
All datasets used for the paper results are available in the [huggingface hub](https://huggingface.co/luckeciano). If you want to generate the datasets by yourself, you can use the following command:


```bash
cd uqlrm/scripts/slurm
sbatch run_reward_inference.sh
```
This command generates the feature datasets for the main results involving the 7B model. For other models, you just need to change the model name accordingly.

# Contact

For any questions or issues, please feel free to contact us via the correspondence e-mail in the paper.   

# Citation

If you find this work helpful in your research, please consider citing:

```
@inproceedings{
    melo2024deep,
    title={Deep Bayesian Active Learning for Preference Modeling in Large Language Models},
    author={Luckeciano C. Melo and Panagiotis Tigas and Alessandro Abate and Yarin Gal},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=TADTT9ughN}
}
```

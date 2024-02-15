# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tqdm import tqdm
from transformers import HfArgumentParser

from datasets import load_dataset
from dataset_utils.dataset_processing_utils import create_datasets, undersample_dataset
from utils import print_trainable_parameters, push_predictions_to_hub
from reward_modeling import (AdapterEnsembleRewardTrainer)
from collators import FrozenBackboneCollator
from configs import RewardConfigWithSavedPredictions
from parsing import AdapterEnsembleRewardModelingArguments
from modules import RewardMLP
import os
import numpy as np
import pandas as pd
from datasets import Dataset

tqdm.pandas()

def build_mini_reward_models(script_args):
    model = RewardMLP(input_size=4096, 
                      layers =script_args.layers,
                      activation_fn=script_args.activation_fn,
                      init_func=script_args.init_func,
                      weight_init=script_args.weight_init)
    return model

def create_datasets(args):

    dataset = load_dataset(
        args.dataset_name,
        #use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None
    )

    train_dataset = dataset[args.train_split]
    valid_dataset = dataset[args.valid_split]
    test_dataset = dataset[args.test_split]
    ood_dataset = dataset[args.ood_split]
    shuffled_dataset = dataset['shuffled']

    return train_dataset, valid_dataset, test_dataset, ood_dataset, shuffled_dataset

parser = HfArgumentParser(AdapterEnsembleRewardModelingArguments)
script_args = parser.parse_args_into_dataclasses()[0]

reward_config = RewardConfigWithSavedPredictions(
            output_dir=script_args.output_dir,
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            per_device_eval_batch_size=script_args.per_device_eval_batch_size,
            num_train_epochs=script_args.num_train_epochs,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            gradient_checkpointing=script_args.gradient_checkpointing,
            learning_rate=script_args.learning_rate,
            report_to=script_args.log_with,
            remove_unused_columns=False,
            warmup_steps=script_args.num_warmup_steps,
            optim=script_args.optimizer_type,
            logging_steps=script_args.logging_steps,
            save_strategy=script_args.save_strategy,
            logging_strategy=script_args.logging_strategy,
            evaluation_strategy=script_args.eval_strategy,
            eval_steps=script_args.eval_steps,
            save_steps=script_args.save_steps,
            save_total_limit=script_args.save_total_limit,
            run_name=script_args.run_name,
            push_predictions_to_hub=script_args.push_predictions_to_hub,
            predictions_dataset_hub=script_args.predictions_dataset_hub,
            save_predictions_steps=script_args.save_predictions_steps,
            lr_scheduler_type=script_args.lr_scheduler_type,
            predictions_dir=os.path.join(script_args.output_dir, "predictions"),
            bf16=script_args.bf16,
            log_level="debug",
            seed=script_args.seed)


model = build_mini_reward_models(script_args)

# Preprocess the dataset and filter out examples that are longer than script_args.max_length
train_dataset, eval_dataset, test_dataset, ood_dataset, shuffled_dataset = create_datasets(script_args)

undersampled_train = undersample_dataset(train_dataset, ratio=0.03, seed=42)
undersampled_eval = undersample_dataset(eval_dataset, ratio=0.1, seed=42)
undersampled_test = undersample_dataset(test_dataset, ratio=0.1, seed=42)
undersampled_ood = undersample_dataset(ood_dataset, ratio=0.95, seed=42)
eval_sets = {"train": undersampled_train, "eval": undersampled_eval, "test": undersampled_test, "ood": undersampled_ood, "shuffled": shuffled_dataset}

# eval_sets = {"train": train_dataset, "eval": eval_dataset, "test": test_dataset, "ood": ood_dataset}

# Define the Trainer
trainer =  AdapterEnsembleRewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=train_dataset,
    eval_dataset=eval_sets,
    data_collator=FrozenBackboneCollator()
)

print_trainable_parameters(model)

if script_args.inference:
    from safetensors.torch import load_file
    from glob import glob
    model_files = glob(f"/users/lucelo/reward-hermes-mlp/vpo_many_equal_final*/checkpoint-*/model.safetensors")
    for idx, model_file in enumerate(model_files):
        loaded_tensors = load_file(model_file)
        model.load_state_dict(loaded_tensors)
        model.eval()
        all_features = []
        all_metadata = []

        for eval_dataset_name, eval_dataset in trainer.eval_dataset.items():
            loss, features = trainer.inference(eval_dataset, return_features=True)

            # features = np.concatenate((features['features_chosen'], features['features_rejected']), axis=1)
            # all_features.extend(features['features_chosen'])
            all_metadata.extend([['chosen', f'{features["rewards_chosen"][i][0]}', f'{script_args.run_name}', f'{eval_dataset_name}', f'{features["id"][i]}'] for i in range(len(features["features_chosen"]))])
            
            # all_features.extend(features['features_rejected'])
            all_metadata.extend([['rejected', f'{features["rewards_rejected"][i][0]}', f'{script_args.run_name}', f'{eval_dataset_name}', f'{features["id"][i]}'] for i in range(len(features["features_rejected"]))])


        # ft_df = pd.DataFrame(all_features).round(4)
        all_metadata_df = pd.DataFrame(all_metadata)

        # ft_df.to_csv(f'features_{script_args.run_name}.tsv', sep='\t', index=False, header=False)
        all_metadata_df.to_csv(f'metadata_{script_args.run_name}_{idx}.tsv', sep='\t', index=False, header=['Preference', 'RewardScore', 'Model', 'Dataset', 'id'])

else:
    trainer.train()

# Upload Predictions to Hub
if script_args.push_predictions_to_hub:
    full_dir = os.path.join(script_args.output_dir, "predictions")
    push_predictions_to_hub(full_dir, script_args.predictions_dataset_hub)



                      
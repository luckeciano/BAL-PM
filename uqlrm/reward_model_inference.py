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
import os
import pandas as pd
import numpy as np
from transformers import HfArgumentParser

from dataset_utils import dataset_process_factory
from dataset_utils.dataset_processing_utils import create_datasets, undersample_dataset

from utils import push_predictions_to_hub
from reward_modeling import (
    RewardDataCollatorWithPaddingAndIndices, RewardInferencer, build_reward_model)
from configs import RewardConfigWithSavedPredictions
from parsing.reward_modeling_parser import RewardModelingArguments


tqdm.pandas()

parser = HfArgumentParser(RewardModelingArguments)
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
            evaluation_strategy=script_args.eval_strategy,
            eval_steps=script_args.eval_steps,
            save_steps=script_args.save_steps,
            max_length=script_args.seq_length,
            run_name=script_args.run_name,
            push_predictions_to_hub=script_args.push_predictions_to_hub,
            predictions_dataset_hub=script_args.predictions_dataset_hub,
            save_predictions_steps=script_args.save_predictions_steps,
            predictions_dir=os.path.join(script_args.output_dir, "predictions"),
            bf16=script_args.bf16,
            log_level="debug")


model, tokenizer, peft_config = build_reward_model(script_args)

# Preprocess the dataset and filter out examples that are longer than script_args.max_length
train_dataset, eval_dataset, test_dataset, ood_dataset = create_datasets(script_args, tokenizer, ood=True)

if script_args.undersample_eval:
    undersampled_train = undersample_dataset(train_dataset, script_args.undersample_ratio, seed=script_args.seed)
    undersampled_eval = undersample_dataset(eval_dataset, script_args.undersample_ratio, seed=script_args.seed)
    undersampled_test = undersample_dataset(test_dataset, script_args.undersample_ratio, seed=script_args.seed)
    undersampled_ood = undersample_dataset(ood_dataset, script_args.undersample_ratio, seed=script_args.seed)
    eval_sets = {"train": undersampled_train, "eval": undersampled_eval, "test": undersampled_test, "ood": undersampled_ood}
else:
    eval_sets = {"train": train_dataset, "eval": eval_dataset, "test": test_dataset, "ood": ood_dataset}

# if script_args.undersample_eval:
#     eval_sets = {"train": undersampled_train, "ood": ood_dataset}

undersampled_train = undersample_dataset(train_dataset, ratio=0.03, seed=script_args.seed)

final_shuffled_test_dataset = undersampled_train.map(lambda example:
    dataset_process_factory.shuffle_tokens(example, tokenizer, reward_config.max_length),
    batched=True,
    num_proc=4,
)

eval_sets['shuffled'] = final_shuffled_test_dataset
    

# Define Reward Collator with Indices:
reward_collator = RewardDataCollatorWithPaddingAndIndices(tokenizer, max_length=reward_config.max_length)

# Define the Trainer
trainer = RewardInferencer(
    model=model,
    tokenizer=tokenizer,
    data_collator=reward_collator,
    args=reward_config,
    train_dataset=train_dataset,
    eval_dataset=eval_sets,
    peft_config=peft_config,
    #bf16=True
)

all_features = []
all_metadata = []

#TODO: Organize this inference code
for eval_dataset_name, eval_dataset in trainer.eval_dataset.items():
    loss, features = trainer.inference(eval_dataset, return_features=True)

    fts = np.concatenate((features['features_chosen'], features['features_rejected']), axis=1)
    ft_df = pd.DataFrame(fts).round(6)
    ft_df['id'] = features['id']
    ft_df = ft_df[['id'] + [col for col in ft_df.columns if col != 'id']]
    ft_df.to_csv(f'features_{eval_dataset_name}_{script_args.run_name}.csv', index=False, header=True)
    
    # all_features.extend(features['features_chosen'])
    # all_metadata.extend([['chosen', f'{features["rewards_chosen"][i][0]}', f'{script_args.run_name}', f'{eval_dataset_name}', f'{features["id"][i]}'] for i in range(len(features["features_chosen"]))])
    
    # all_features.extend(features['features_rejected'])
    # all_metadata.extend([['rejected', f'{features["rewards_rejected"][i][0]}', f'{script_args.run_name}', f'{eval_dataset_name}', f'{features["id"][i]}'] for i in range(len(features["features_rejected"]))])


# ft_df = pd.DataFrame(all_features).round(4)
# all_metadata_df = pd.DataFrame(all_metadata)

# ft_df.to_csv(f'features_{script_args.run_name}.tsv', sep='\t', index=False, header=False)
# all_metadata_df.to_csv(f'metadata_{script_args.run_name}.tsv', sep='\t', index=False, header=['Preference', 'RewardScore', 'Model', 'Dataset', 'id'])

    
    

# Upload Predictions to Hub
if script_args.push_predictions_to_hub:
    eval_sets = {"train": train_dataset, "eval": eval_dataset, "test": test_dataset, "ood": ood_dataset}

    for eval_dataset_name, eval_dataset in eval_sets.items():
        dataset_metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=f"eval_{eval_dataset_name}",
        )

    full_dir = os.path.join(script_args.output_dir, "predictions")
    push_predictions_to_hub(full_dir, script_args.predictions_dataset_hub)
    print("done")
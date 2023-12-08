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

from dataset_utils import dataset_process_factory
from dataset_utils.dataset_processing_utils import create_datasets, undersample_dataset

from utils import print_trainable_parameters, compute_accuracy_with_inputs, EvaluateFirstStepCallback
from reward_modeling import (
    RewardDataCollatorWithPaddingAndIndices, RewardTrainerWithCustomEval, build_reward_model)
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
            bf16=script_args.bf16,
            log_level="debug")


model, tokenizer, peft_config = build_reward_model(script_args)

# Preprocess the dataset and filter out examples that are longer than script_args.max_length
train_dataset, eval_dataset, test_dataset, ood_dataset = create_datasets(script_args, reward_config, tokenizer, ood=True)

if script_args.undersample_eval:
    undersampled_train = undersample_dataset(train_dataset, script_args.undersample_ratio)
    undersampled_eval = undersample_dataset(eval_dataset, script_args.undersample_ratio)
    undersampled_test = undersample_dataset(test_dataset, script_args.undersample_ratio)
    eval_sets = {"train": undersampled_train, "eval": undersampled_eval, "test": undersampled_test, "ood": ood_dataset}
else:
    eval_sets = {"train": train_dataset, "eval": eval_dataset, "test": test_dataset, "ood": ood_dataset}

# Adding a shuffled version of the test dataset
final_shuffled_test_dataset = eval_sets['test'].map(lambda example:
    dataset_process_factory.shuffle_tokens(example, tokenizer, reward_config.max_length),
    batched=True,
    num_proc=4,
)

eval_sets['shuffled'] = final_shuffled_test_dataset

# Define Reward Collator with Indices:
reward_collator = RewardDataCollatorWithPaddingAndIndices(tokenizer, max_length=reward_config.max_length)

# Define the Trainer
trainer = RewardTrainerWithCustomEval(
    model=model,
    tokenizer=tokenizer,
    data_collator=reward_collator,
    args=reward_config,
    train_dataset=train_dataset,
    eval_dataset=eval_sets,
    peft_config=peft_config,
    #bf16=True
)

trainer.add_callback(EvaluateFirstStepCallback())

print_trainable_parameters(model)

trainer.train()
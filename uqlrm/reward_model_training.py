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
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, AutoModelForCausalLM

from dataset_utils import dataset_process_factory
from utils import print_trainable_parameters, compute_accuracy_with_inputs, EvaluateFirstStepCallback
from reward_modeling import (
    RewardDataCollatorWithPaddingAndIndices, RewardConfigWithSavedPredictions, RewardTrainerWithCustomEval)
from parsing.reward_modeling_parser import RewardModelingArguments


tqdm.pandas()

def build_reward_model(script_args):
    # Load the model
    device_map = {"": Accelerator().local_process_index}
    torch_dtype = torch.bfloat16

    if script_args.quantization_scheme == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif script_args.quantization_scheme == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        print("Loading model with no quantization!")
        device_map = None
        bnb_config = None
        torch_dtype = None
    
    
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype=torch_dtype
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True, truncation=True, max_length=script_args.seq_length)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    model.config.pad_token_id = model.config.eos_token_id # fix
    
    return model, tokenizer

def process_and_filter_dataset(dataset, reward_config, tokenizer):
    final_dataset = dataset.map(lambda example:
        getattr(dataset_process_factory, script_args.preprocess_fn)(example, tokenizer, reward_config.max_length),
        batched=True,
        num_proc=4,
    )
    final_dataset = final_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        and len(x["input_ids_rejected"]) <= reward_config.max_length
    )
    print(f"Size of the set before processing: {len(dataset)}, after processing: {len(final_dataset)}")
    return final_dataset

def create_datasets(args, reward_config, tokenizer, ood=False):
    dataset = load_dataset(
        args.dataset_name,
        #use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None
    )

    train_dataset = dataset[args.train_split]
    valid_dataset = dataset[args.valid_split]
    test_dataset = dataset[args.test_split]

    final_train_dataset = process_and_filter_dataset(train_dataset, reward_config, tokenizer)
    final_valid_dataset = process_and_filter_dataset(valid_dataset, reward_config, tokenizer)
    final_test_dataset = process_and_filter_dataset(test_dataset, reward_config, tokenizer)

    if ood:
        ood_dataset = dataset[args.ood_split]
        final_ood_dataset = process_and_filter_dataset(ood_dataset, reward_config, tokenizer)
        return final_train_dataset, final_valid_dataset, final_test_dataset, final_ood_dataset

    return final_train_dataset, final_valid_dataset, final_test_dataset, final_shuffled_test_dataset

def undersample_dataset(dataset, ratio):
    dataset = dataset.train_test_split(test_size=ratio, seed=42)
    return dataset["test"]


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


if script_args.use_peft:
    peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            target_modules=script_args.peft_lora_target_modules,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        )
else:
    peft_config = None

model, tokenizer = build_reward_model(script_args)

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
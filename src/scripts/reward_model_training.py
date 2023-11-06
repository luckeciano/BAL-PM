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
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser

from trl import RewardConfig, RewardTrainer


tqdm.pandas()

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    run_name: Optional[str] = field(default="rwft_opt350", metadata={"help": "The experiment name"})

    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "Dataset text column name"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    test_split_size: Optional[float] = field(default=0.005, metadata={"help": "size of test split"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    max_steps: Optional[int] = field(default=-1, metadata={"help": "Max gradient steps. Overrides num_train_epochs if set."})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "number of epochs"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the logging frequency"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "frequency of validation eval"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "evaluation strategy"})
    save_steps: Optional[int] = field(default=5000, metadata={"help": "the saving frequency"})
    per_device_train_batch_size: Optional[int] = field(default=64, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=16, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=16, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    peft_lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "the dropout parameter of the LoRA adapters"})
    peft_lora_target_modules: Optional[List[str]] = field(default=None, metadata={"help": "target modules of the LoRA adapters"})
    quantization_scheme: Optional[str] = field(default="none", metadata={"help": "quantization scheme for the LLM (8bit, 4bit, none)"})
    no_model_cache: Optional[bool] = field(default=False, metadata={"help": "Disable model cache to save VRAM"})

    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
reward_config = RewardConfig(
            output_dir=script_args.output_dir,
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            num_train_epochs=script_args.num_train_epochs,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            gradient_checkpointing=script_args.gradient_checkpointing,
            learning_rate=script_args.learning_rate,
            report_to=script_args.log_with,
            remove_unused_columns=False,
            optim=script_args.optimizer_type,
            logging_steps=script_args.log_freq,
            evaluation_strategy=script_args.eval_strategy,
            max_length=script_args.seq_length,
            run_name=script_args.run_name,
            log_level="debug")


use_peft: bool = script_args.use_peft

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


# Step 2: Load the dataset and pre-process it
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True, truncation=True, max_length=script_args.seq_length)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
model.config.pad_token_id = model.config.eos_token_id # fix

def create_datasets(args):
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        #use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=args.test_split_size, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]

        train_dataset = train_data.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
            and len(x["input_ids_rejected"]) <= reward_config.max_length
        )

        eval_dataset = valid_data.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
            and len(x["input_ids_rejected"]) <= reward_config.max_length
        )

        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    return train_dataset, eval_dataset


# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


# Preprocess the dataset and filter out examples that are longer than script_args.max_length
train_dataset, eval_dataset = create_datasets(script_args)

# Define the Trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=reward_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    #bf16=True
)

trainer.train()
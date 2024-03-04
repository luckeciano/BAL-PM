import os
from dataclasses import dataclass, field
from typing import Optional, Union, List
from tqdm import tqdm
from utils import print_trainable_parameters

import torch

from datasets import load_dataset
from dataset_utils import dataset_process_factory, process_and_filter_dataset
from dataset_utils.dataset_processing_utils import undersample_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    HfArgumentParser, TrainingArguments, BitsAndBytesConfig, logging

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

logger = logging.get_logger(__name__)

# dataset = load_dataset("webis/tldr-17", split="train")
# device = "cuda"
# model_id = "gpt2"
# gpt2 = GPT2LMHeadModel.from_pretrained(model_id).to(device)


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    run_name: Optional[str] = field(default="SFT_default", metadata={"help": "The experiment name"})

    dataset_name: Optional[str] = field(default="webis/tldr-17", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(default="normalizedBody", metadata={"help": "Dataset text column name"})
    train_split: Optional[str] = field(default="train", metadata={"help": "the train split to use"})
    valid_split: Optional[str] = field(default="valid1", metadata={"help": "the validation split to use"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    preprocess_fn: Optional[str] = field(default="process_reddit_sft", metadata={"help": "dataset preprocess function"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    test_split_size: Optional[float] = field(default=0.005, metadata={"help": "size of test split"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    max_steps: Optional[int] = field(default=-1, metadata={"help": "Max gradient steps. Overrides num_train_epochs if set."})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "number of epochs"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the logging frequency"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "frequency of validation eval"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "evaluation strategy"})
    save_steps: Optional[int] = field(default=5000, metadata={"help": "the saving frequency"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    undersample_eval: Optional[bool] = field(default=False, metadata={"help": "whether to undersample eval datasets for faster evaluation"})
    undersample_ratio: Optional[float] = field(default=0.1, metadata={"help": "ratio of the dataset to consider for faster eval"})

    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    peft_lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "the dropout parameter of the LoRA adapters"})
    peft_lora_target_modules: Optional[List[str]] = field(default=None, metadata={"help": "target modules of the LoRA adapters"})
    quantization_scheme: Optional[str] = field(default="4bit", metadata={"help": "quantization scheme for the LLM (8bit, 4bit, none)"})
    no_model_cache: Optional[bool] = field(default=False, metadata={"help": "Disable model cache to save VRAM"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    seed: Optional[int] = field(default=42, metadata={"help": "Experiment run seed."})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return example[script_args.dataset_text_field]

def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        #use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    elif args.valid_split is not None:
        train_data = dataset[args.train_split]
        valid_data = dataset[args.valid_split]
        if script_args.undersample_eval:
            valid_data = undersample_dataset(valid_data, script_args.undersample_ratio, seed=script_args.seed)

        train_data = train_data.map(lambda example:
            getattr(dataset_process_factory, script_args.preprocess_fn)(example, tokenizer, script_args.seq_length),
            batched=True,
            num_proc=4,
        )

        valid_data = valid_data.map(lambda example:
            getattr(dataset_process_factory, script_args.preprocess_fn)(example, tokenizer, script_args.seq_length),
            batched=True,
            num_proc=4,
        )
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    else:
        dataset = dataset[args.train_split]
        dataset = dataset.train_test_split(test_size=args.test_split_size, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print("Splitting dataset into train/test splits...")
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset

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

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    # use_auth_token=True,
)


if script_args.no_model_cache:
    base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True, truncation=True, max_length=script_args.seq_length)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Define Training Parameters
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    evaluation_strategy=script_args.eval_strategy,
    eval_steps=script_args.eval_steps,
    num_train_epochs=script_args.num_train_epochs,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    bf16=True,
    remove_unused_columns=False,
    run_name=script_args.run_name,
    log_level="debug"
)

# Load Dataset
train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

# Define the Lora Config
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=script_args.peft_lora_dropout,
        target_modules=script_args.peft_lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    print("Disabling PEFT...")
    peft_config = None

# Define the Trainer
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    max_seq_length=script_args.seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

print_trainable_parameters(trainer.model)
trainer.train()


trainer.save_model(script_args.output_dir)
output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

if script_args.use_peft:
    # Free memory for merging weights
    del base_model
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class RewardModelingArguments:
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="gpt2", metadata={"help": "the tokenizer name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    run_name: Optional[str] = field(default="rwft_opt350", metadata={"help": "The experiment name"})

    dataset_name: Optional[str] = field(default="luckeciano/learning-to-summarize", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "Dataset text column name"})
    preprocess_fn: Optional[str] = field(default="redditcnn_preprocess_function", metadata={"help": "Which preprocess fn to apply"})
    train_split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    valid_split: Optional[str] = field(default="valid1", metadata={"help": "the split to use"})
    test_split: Optional[str] = field(default="valid2_reddit", metadata={"help": "the split to use"})
    ood_split: Optional[str] = field(default="valid2_cnn", metadata={"help": "the split to use"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    test_split_size: Optional[float] = field(default=0.005, metadata={"help": "size of test split"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    pin_memory: Optional[bool] = field(default=True, metadata={"help": "dataloader pin memory"})
    undersample_eval: Optional[bool] = field(default=False, metadata={"help": "whether to undersample eval datasets for faster evaluation"})
    undersample_ratio: Optional[float] = field(default=0.1, metadata={"help": "ratio of the dataset to consider for faster eval"})

    max_steps: Optional[int] = field(default=-1, metadata={"help": "Max gradient steps. Overrides num_train_epochs if set."})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "number of epochs"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the logging frequency"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "frequency of validation eval"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "evaluation strategy"})
    save_steps: Optional[int] = field(default=5000, metadata={"help": "the saving frequency"})
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "Max number of checkpoints per member."})
    per_device_train_batch_size: Optional[int] = field(default=64, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=64, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=16, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    bf16: Optional[bool] = field(default=True, metadata={"help": "whether to enable bf16 training"})

    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=16, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    peft_lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "the dropout parameter of the LoRA adapters"})
    peft_lora_target_modules: Optional[List[str]] = field(default=None, metadata={"help": "target modules of the LoRA adapters"})
    quantization_scheme: Optional[str] = field(default="none", metadata={"help": "quantization scheme for the LLM (8bit, 4bit, none)"})
    no_model_cache: Optional[bool] = field(default=False, metadata={"help": "Disable model cache to save VRAM"})

    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=5, metadata={"help": "the number of warmup steps"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})

    push_predictions_to_hub: Optional[bool] = field(default=False, metadata={"help": "Enable storing predictions from test sets in hub."})
    predictions_dataset_hub: Optional[str] = field(default=None, metadata={"help": "The datasets hub repository to save predictions."})
    save_predictions_steps: Optional[int] = field(default=20, metadata={"help": "the saving predictions frequency"})

    seed: Optional[int] = field(default=42, metadata={"help": "Experiment run seed."})
    
    inference: Optional[bool] = field(default=False, metadata={"help": "whether it is running inference only"})
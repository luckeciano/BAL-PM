from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AdapterEnsembleRewardModelingArguments:
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    run_name: Optional[str] = field(default="rwft_opt350", metadata={"help": "The experiment name"})

    dataset_name: Optional[str] = field(default="luckeciano/reddit-features-hermes-mini", metadata={"help": "the dataset name"})
    train_split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    valid_split: Optional[str] = field(default="eval", metadata={"help": "the split to use"})
    test_split: Optional[str] = field(default="test", metadata={"help": "the split to use"})
    ood_split: Optional[str] = field(default="ood", metadata={"help": "the split to use"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    test_split_size: Optional[float] = field(default=0.005, metadata={"help": "size of test split"})
    undersample_eval: Optional[bool] = field(default=False, metadata={"help": "whether to undersample eval datasets for faster evaluation"})
    undersample_ratio: Optional[float] = field(default=0.1, metadata={"help": "ratio of the dataset to consider for faster eval"})
    num_workers: Optional[int] = field(default=10, metadata={"help": "the number of workers"})

    max_steps: Optional[int] = field(default=-1, metadata={"help": "Max gradient steps. Overrides num_train_epochs if set."})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "number of epochs"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the logging frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "frequency of validation eval"})
    eval_strategy: Optional[str] = field(default="epoch", metadata={"help": "evaluation strategy"})
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "evaluation strategy"})
    logging_strategy: Optional[str] = field(default="epoch", metadata={"help": "evaluation strategy"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    save_predictions_steps: Optional[int] = field(default=1, metadata={"help": "the saving predictions frequency"})
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "Max number of checkpoints per member."})
    per_device_train_batch_size: Optional[int] = field(default=1024, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1024, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    bf16: Optional[bool] = field(default=True, metadata={"help": "whether to enable bf16 training"})

    no_model_cache: Optional[bool] = field(default=False, metadata={"help": "Disable model cache to save VRAM"})

    learning_rate: Optional[float] = field(default=3e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=0, metadata={"help": "the number of warmup steps"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})

    push_predictions_to_hub: Optional[bool] = field(default=False, metadata={"help": "Enable storing predictions from test sets in hub."})
    predictions_dataset_hub: Optional[str] = field(default="luckeciano/uqlrm_predictions", metadata={"help": "The datasets hub repository to save predictions."})
    

    # MLP arguments
    layers: Optional[str] = field(default='[2048, 256]', metadata={"help": "mlp layers"})
    activation_fn: Optional[str] = field(default='tanh', metadata={"help": "activation function"})
    init_func: Optional[str] = field(default='normal', metadata={"help": "weight init scheme"})
    weight_init: Optional[float] = field(default=0.01, metadata={"help": "weight init dispersion parameter"})

    seed: Optional[int] = field(default=42, metadata={"help": "Experiment run seed."})
    inference: Optional[bool] = field(default=False, metadata={"help": "whether it is running inference only"})
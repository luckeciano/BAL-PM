from huggingface_hub import HfApi
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    checkpoint_folder: Optional[str] = field(default=None, metadata={"help": "the model name"})
    repo_id: Optional[str] = field(default=None, metadata={"help": "the model name"})
    repo_type: Optional[str] = field(default="model", metadata={"help": "model, dataset, space"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

api = HfApi()
api.upload_folder(
    folder_path=script_args.checkpoint_folder,
    repo_id=script_args.repo_id,
    repo_type=script_args.repo_type,
)

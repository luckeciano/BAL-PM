from huggingface_hub import HfApi
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    filepath: Optional[str] = field(default=None, metadata={"help": "the model name"})
    repo_id: Optional[str] = field(default=None, metadata={"help": "the model name"})
    path_in_repo: Optional[str] = field(default="model", metadata={"help": "path in repository"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

api = HfApi()
api.upload_file(
    path_or_fileobj=script_args.filepath,
    path_in_repo=script_args.path_in_repo,
    repo_id=script_args.repo_id,
    repo_type="dataset"
)
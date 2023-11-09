from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers import  PreTrainedTokenizerBase, BatchEncoding
from trl.trainer.utils import RewardDataCollatorWithPadding

@dataclass
class RewardDataCollatorWithPaddingAndIndices(RewardDataCollatorWithPadding):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        indices = {"id": []}
        for feature in features:
            indices["id"].append(feature["id"])

        ids_tensor = BatchEncoding.convert_to_tensors(indices, self.return_tensors)
        batch["id"] = ids_tensor["id"]
        return batch
from typing import Any, Dict, List, Mapping, Optional, Union, Tuple, Callable
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from reward_modeling import RewardTrainerWithCustomEval
from accelerate import Accelerator

from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerCallback
from trl.trainer import RewardConfig
from transformers.trainer_utils import (
    EvalPrediction,
)

import numpy as np
import gc


class RewardInferencer:
    r"""
    This Reward Trainer Class implements a custom evaluation step.
    This evaluation runs on the beginning of the training (as a callback)
    and saves the predictions in an external file.
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: Optional[RewardConfig] = None,
            data_collator: Optional[DataCollator] = None,
        ):
        self.data_collator = data_collator
        self.args = args
        self.model = model
        self.accelerator = Accelerator()

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return None
            
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        return eval_dataloader
    
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            return data.to(**kwargs)
        return data
    
    def inference(self,
        eval_dataset: Optional[Dataset] = None,
        return_features: Optional[bool] = False,
        filepath_fts: Optional[str] = None,
        filepath_chosen: Optional[str] = None,
    ) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        _is_quantized = getattr(self.model, "is_quantized", False)
        model = self.model if _is_quantized else self.model.to(self.args.device)
        
        model.eval()

        model, eval_dataloader = self.accelerator.prepare(model, eval_dataloader)
        with torch.no_grad():
            for step, inputs in tqdm(enumerate(eval_dataloader)):
                inputs = self._prepare_input(inputs)

                ids = inputs['id'].float().cpu().numpy().reshape(-1, 1)

                output_chosen = model(
                    input_ids=inputs["input_ids_chosen"],
                    attention_mask=inputs["attention_mask_chosen"],
                    output_hidden_states=True
                )
                # output_rejected = model(
                #     input_ids=inputs["input_ids_rejected"],
                #     attention_mask=inputs["attention_mask_rejected"],
                #     output_hidden_states=True
                # )

                # rewards_chosen = output_chosen[0]
                # rewards_rejected = output_rejected[0]
                # calculate loss, optionally modulate with margin
                # if "margin" in inputs:
                #     loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
                # else:
                #     loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()



                inputs_chosen = inputs['input_ids_chosen']
                # inputs_rejected = inputs['input_ids_rejected']

                batch_size, _ = inputs_chosen.shape[:2]

                assert (
                    model.config.pad_token_id is not None or batch_size == 1
                ), "Cannot handle batch sizes > 1 if no padding token is defined."

                
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths_chosen = torch.eq(inputs_chosen, model.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths_chosen = sequence_lengths_chosen % inputs_chosen.shape[-1]
                # sequence_lengths = sequence_lengths.to(logits.device)

                # sequence_lengths_rejected = torch.eq(inputs_rejected, model.config.pad_token_id).int().argmax(-1) - 1
                # sequence_lengths_rejected = sequence_lengths_rejected % inputs_rejected.shape[-1]


                features_chosen = output_chosen.hidden_states[-1][torch.arange(batch_size), sequence_lengths_chosen, :]
                # features_rejected = output_rejected.hidden_states[-1][torch.arange(batch_size), sequence_lengths_rejected, :]
                
                features_chosen = features_chosen.float().cpu().numpy()
                # features_rejected = features_rejected.float().cpu().numpy()
                

                # fts = np.concatenate((ids, features_chosen, features_rejected), axis=1)
                fts_chosen = np.concatenate((ids, features_chosen), axis=1)

                if step == 0:
                    # np.savetxt(filepath_fts, fts, delimiter=',', fmt='%.5f')
                    np.savetxt(filepath_chosen, fts_chosen, delimiter=',', fmt='%.5f')
                else:
                    # with open(filepath_fts, 'ab') as f:
                    #     np.savetxt(f, fts, delimiter=',', fmt='%.5f')
                    with open(filepath_chosen, 'ab') as f:
                        np.savetxt(f, fts_chosen, delimiter=',', fmt='%.5f')
                
                del sequence_lengths_chosen, features_chosen, ids, output_chosen
                gc.collect()
                torch.cuda.empty_cache()

        
        
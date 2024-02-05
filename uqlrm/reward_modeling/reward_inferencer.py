from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
from reward_modeling import RewardTrainerWithCustomEval

from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerCallback
from trl.trainer.training_configs import RewardConfig
from transformers.trainer_utils import (
    EvalPrediction,
)


class RewardInferencer(RewardTrainerWithCustomEval):
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
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
                None,
                None,
            ),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            max_length: Optional[int] = None,
            peft_config: Optional[Dict] = None
        ):
            self.main_input_name = "id"
            self.predictions = {}
            self.run_dir = ""
            super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, \
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics, max_length, peft_config)
            
    def inference(self,
        eval_dataset: Optional[Dataset] = None,
        return_features: Optional[bool] = False,
    ) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
        
        model.eval()

        all_features_chosen = []
        all_features_rejected = []
        all_rewards_chosen = []
        all_rewards_rejected = []
        all_ids = []
        for step, inputs in tqdm(enumerate(eval_dataloader)):
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                output_chosen = model(
                    input_ids=inputs["input_ids_chosen"],
                    attention_mask=inputs["attention_mask_chosen"],
                    output_hidden_states=True
                )
                output_rejected = model(
                    input_ids=inputs["input_ids_rejected"],
                    attention_mask=inputs["attention_mask_rejected"],
                    output_hidden_states=True
                )

            rewards_chosen = output_chosen[0]
            rewards_rejected = output_rejected[0]
            # calculate loss, optionally modulate with margin
            if "margin" in inputs:
                loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
            else:
                loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()



            inputs_chosen = inputs['input_ids_chosen']
            inputs_rejected = inputs['input_ids_rejected']

            batch_size, sequence_length = inputs_chosen.shape[:2]

            assert (
                model.config.pad_token_id is not None or batch_size == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."

            
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths_chosen = torch.eq(inputs_chosen, model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths_chosen = sequence_lengths_chosen % inputs_chosen.shape[-1]
            # sequence_lengths = sequence_lengths.to(logits.device)

            sequence_lengths_rejected = torch.eq(inputs_rejected, model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths_rejected = sequence_lengths_rejected % inputs_rejected.shape[-1]


            features_chosen = output_chosen.hidden_states[-1][torch.arange(batch_size), sequence_lengths_chosen, :]
            features_rejected = output_rejected.hidden_states[-1][torch.arange(batch_size), sequence_lengths_rejected, :]
            

            all_rewards_chosen.extend(rewards_chosen.cpu().numpy())
            all_rewards_rejected.extend(rewards_rejected.cpu().numpy())
            all_features_chosen.extend(features_chosen.cpu().numpy())
            all_features_rejected.extend(features_rejected.cpu().numpy())
            all_ids.extend(inputs['id'].cpu().numpy())
        if return_features:
            return loss, {
                "rewards_chosen": all_rewards_chosen,
                "rewards_rejected": all_rewards_rejected,
                "features_chosen": all_features_chosen,
                "features_rejected": all_features_rejected,
                "id": all_ids
            }
        return loss
    
    
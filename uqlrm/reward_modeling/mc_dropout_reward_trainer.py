from trl import RewardTrainer
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from metrics import compute_uncertanties
from reward_modeling import AdapterEnsembleRewardTrainer
from scipy.stats import norm

import warnings

from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations.deepspeed import deepspeed_init
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerCallback
from trl.trainer import RewardConfig
from transformers.trainer_utils import (
    EvalPrediction,
)

from transformers.utils import (
    logging
)


logger = logging.get_logger(__name__)

class MCDropoutRewardTrainer(AdapterEnsembleRewardTrainer):
    r"""
    This Reward Trainer Class trains an  MC Dropout reward model for Uncertainty Estimation.
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
            peft_config: Optional[Dict] = None,
        ):  
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

        all_preferences = []
        for _ in range(self.args.mc_dropout_realizations):
            realization_prefs = []
            realization_ids = []
            model.freeze()
            for step, inputs in tqdm(enumerate(eval_dataloader)):
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False)
                
                preferences = logits[:, 0]
                realization_prefs.extend(preferences.cpu().numpy())
                realization_ids.extend(inputs['id'].cpu().numpy())

            model.unfreeze()
            all_preferences.append({"preferences": realization_prefs,"id": realization_ids})

        if return_features:
            return loss, all_preferences
        return loss
    
    def compute_uncertainties(self, runs, mode, all_preds):
        assert len(runs) == 1, "Only one model is required for MC Dropout"
        run = runs[0]
        realizations_dfs = []
        for realization in all_preds[run.run_name][f'eval_{mode}']:
            realizations_dfs.append(realization[['First', 'Second']].to_numpy())
            ids = realization[['id']]
        
        return compute_uncertanties(realizations_dfs, ids)
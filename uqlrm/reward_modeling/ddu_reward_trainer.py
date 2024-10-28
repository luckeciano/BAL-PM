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

class DDURewardTrainer(AdapterEnsembleRewardTrainer):
    r"""
    This Reward Trainer Class DDU on top of the reward latent for Uncertainty Estimation.
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

    def compute_uncertainties(self, runs, mode, all_preds):

        assert len(runs) == 1, "Only one model is required for DDU"
        run = runs[0]

        rewards_chosen = all_preds[run.run_name][f'eval_buffer']['reward_chosen']
        rewards_rejected = all_preds[run.run_name][f'eval_buffer']['reward_rejected']  

        chosen_gd = norm(loc=rewards_chosen.mean(), scale=rewards_chosen.std())
        rejected_gd = norm(loc=rewards_rejected.mean(), scale=rewards_rejected.std())

        train_rw_chosen = all_preds[run.run_name][f'eval_{mode}']['reward_chosen']
        train_rw_rejected = all_preds[run.run_name][f'eval_{mode}']['reward_rejected']
        prob_chosen = all_preds[run.run_name][f'eval_{mode}']['First']
        prob_rejected = all_preds[run.run_name][f'eval_{mode}']['Second']
        ids = all_preds[run.run_name][f'eval_{mode}']['id']

        chosen_density = chosen_gd.cdf(train_rw_chosen)
        rejected_density = rejected_gd.cdf(train_rw_rejected)

        chosen_alpha = np.where(chosen_density < 0.5, chosen_density, 1 - chosen_density)
        rejected_alpha = np.where(rejected_density < 0.5, rejected_density, 1 - rejected_density)

        uncertainty = - (chosen_alpha + rejected_alpha) # "Averaging" alphas and flipping sign for descendence order
        # uncertainty = - np.minimum(chosen_alpha, rejected_alpha)

        df = pd.DataFrame({'Epistemic Uncertainty': uncertainty, 'Predictive Uncertainty': np.zeros(len(uncertainty)), 
                        'Aleatoric Uncertainty': np.zeros(len(uncertainty)), 'First': prob_chosen, 'Second': prob_rejected,
                        'Variance': np.zeros(len(uncertainty)), 'id': ids })

        return df['Epistemic Uncertainty'], df['Predictive Uncertainty'], df['Aleatoric Uncertainty'], df[['First', 'Second']], df['Variance'], df['id']
    

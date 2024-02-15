from . import RewardTrainerWithCustomEval
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import pandas as pd
import math
import numpy as np
from tqdm import tqdm

import warnings

from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations.deepspeed import deepspeed_init
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerCallback
from trl.trainer.training_configs import RewardConfig
from transformers.trainer_utils import (
    EvalPrediction,
)

from transformers.utils import (
    is_torch_tpu_available,
    is_accelerate_available,
    logging
)

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


logger = logging.get_logger(__name__)

class VariationalRewardTrainer(RewardTrainerWithCustomEval):
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
            self._likelihood_loss = []
            self._last_logged = 0
            super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, \
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics, max_length, peft_config)
            
    
    def _kl_beta(self, kl_annealing):
        for item in kl_annealing:
            yield item

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        # self._kl_annealing = self.kl_annealing(pre_annealing_steps=1000, start=0.0, stop=0.1, step=0.0001, n_steps=num_training_steps)
        # self.kl_beta_func = self._kl_beta(self._kl_annealing)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )

        inputs_chosen = inputs['input_chosen']
        inputs_rejected = inputs['input_rejected']

        rewards_chosen, mu_chosen, logvar_chosen = model(
            inputs_chosen
        )
        rewards_rejected, mu_rejected, logvar_rejected = model(
            inputs_rejected
        )

        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        kl_div_chosen = - 0.5 * torch.mean(1+ logvar_chosen - mu_chosen.pow(2) - logvar_chosen.exp())
        kl_div_rejected = - 0.5 * torch.mean(1+ logvar_rejected - mu_rejected.pow(2) - logvar_rejected.exp())

        # KL Annealing
        #beta = next(self.kl_beta_func)
        beta = 0.05

        final_loss = loss + beta * kl_div_chosen + beta * kl_div_rejected

        self._likelihood_loss.append(loss.detach().cpu().numpy().tolist())
        self.kl_beta = beta
        self.tr_kl_chosen = kl_div_chosen.detach().cpu().numpy().tolist()
        self.tr_kl_rejected = kl_div_rejected.detach().cpu().numpy().tolist()
        self.exp_var_chosen = torch.mean((0.5 * logvar_chosen).exp()).detach().cpu().numpy().tolist()
        self.exp_var_rejected = torch.mean((0.5 * logvar_rejected).exp()).detach().cpu().numpy().tolist()

        if return_outputs:
            return final_loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected
            }
        return final_loss
    
    def kl_annealing_cycle_linear(self, start, stop, n_steps, n_cycle=4, ratio=0.5):
        L = stop * np.ones(n_steps)
        period = n_steps/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):

            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_steps):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L    


    def kl_annealing_cycle_sigmoid(self, start, stop, n_steps, n_cycle=4, ratio=0.5):
        L = np.ones(n_steps)
        period = n_steps/n_cycle
        step = (stop-start)/(period*ratio) # step is in [0,1]
        
        # transform into [-6, 6] for plots: v*12.-6.

        for c in range(n_cycle):

            v , i = start , 0
            while v <= stop:
                L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
                v += step
                i += 1
        return L    


    #  function  = 1 − cos(a), where a scans from 0 to pi/2

    def kl_annealing_cycle_cosine(self, start, stop, n_steps, n_cycle=4, ratio=0.5):
        L = np.ones(n_steps)
        period = n_steps/n_cycle
        step = (stop-start)/(period*ratio) # step is in [0,1]
        
        # transform into [0, pi] for plots: 

        for c in range(n_cycle):

            v , i = start , 0
            while v <= stop:
                L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
                v += step
                i += 1
        return L    
    
    def kl_annealing(self, pre_annealing_steps, start, stop, step, n_steps):
        L = stop * np.ones(n_steps)
        v , i = start , 0
        while v <= stop:
            if i < pre_annealing_steps:
                L[i] = 0.0
            else:
                L[i] = v
                v += step
            i += 1
        return L
    

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """

        num_steps = self.state.global_step - self._last_logged
        #TODO Fix for num_steps = 0
        logs['likelihood_loss'] = np.mean(self._likelihood_loss)
        self._likelihood_loss =  [] #reset loss
        logs['kl_beta'] = self.kl_beta
        logs['kl_div_chosen'] = self.tr_kl_chosen
        logs['kl_div_rejected'] = self.tr_kl_rejected
        logs['var_chosen'] = self.exp_var_chosen
        logs['var_rejected'] = self.exp_var_rejected

        super().log(logs)

    def inference(self,
        eval_dataset: Optional[Dataset] = None
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

        all_mu_chosen = []
        all_mu_rejected = []
        all_var_chosen = []
        all_var_rejected = []
        all_ids = []
        for step, inputs in tqdm(enumerate(eval_dataloader)):
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                inputs_chosen = inputs['input_chosen']
                inputs_rejected = inputs['input_rejected']

                rewards_chosen, mu_chosen, logvar_chosen = model(
                    inputs_chosen
                )
                rewards_rejected, mu_rejected, logvar_rejected = model(
                    inputs_rejected
                )
            
                var_chosen = torch.exp(0.5 * logvar_chosen)
                var_rejected = torch.exp(0.5 * logvar_rejected)

                all_mu_chosen.extend(mu_chosen.cpu().numpy())
                all_mu_rejected.extend(mu_rejected.cpu().numpy())
                all_var_chosen.extend(var_chosen.cpu().numpy())
                all_var_rejected.extend(var_rejected.cpu().numpy())
                all_ids.extend(inputs['id'].cpu().numpy())

        return {
            "mu_chosen": all_mu_chosen,
            "mu_rejected": all_mu_rejected,
            "logvar_chosen": all_var_chosen,
            "logvar_rejected": all_var_rejected,
            "id": all_ids
        }


    
from trl import RewardTrainer
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import pandas as pd
import copy
import numpy as np
from packaging import version
from tqdm import tqdm
from metrics import compute_uncertanties

import warnings

from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations.deepspeed import deepspeed_init
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerCallback
from trl.trainer import RewardConfig
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length
)

from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify
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

class AdapterEnsembleRewardTrainer(RewardTrainer):
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
            peft_config: Optional[Dict] = None,
        ):
            self.main_input_name = "id"
            self.predictions = {}
            self.run_dir = ""
            self.regularized_loss = args.regularized_loss
            
            super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, \
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics, max_length, peft_config)
            
            if self.regularized_loss:
                self.base_parameters = [copy.deepcopy(p.data).detach() for p in model.parameters()]
                self.lambda_reg = args.lambda_regularization

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

        rewards_chosen = model(
            inputs_chosen
        )
        rewards_rejected = model(
            inputs_rejected
        )

        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if self.regularized_loss:
            curr_params = [p.data for p in model.parameters()]
            l2_dist = sum((p1 - p2).norm(2).item() for p1, p2 in zip(curr_params, self.base_parameters))
            reg = self.lambda_reg / self.state.max_steps # Consider the number of gradient steps in the regularizer

            loss = loss + reg * l2_dist

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs[self.main_input_name])

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and (self.accelerator.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Save Predictions
        self.save_predictions(all_preds, all_inputs, None, args.run_name, metric_key_prefix)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    
    def save_predictions(self, predictions, indices, trial, run_name, metric_key_prefix="eval"):
        if self.state.global_step % self.args.save_predictions_steps == 0 or self.state.global_step == 1:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            output_dir = os.path.join(self.args.predictions_dir, run_name, checkpoint_folder, metric_key_prefix)
            df = pd.DataFrame(predictions, columns=["First", "Second"])
            df['id'] = indices.reshape(-1, 1)
            
            self.predictions[metric_key_prefix] = df

            # Create the folder if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_filedir=os.path.join(output_dir, "predictions.csv")
            df.to_csv(output_filedir, index=False)

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
        all_preferences = []
        all_ids = []
        for step, inputs in tqdm(enumerate(eval_dataloader)):
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                inputs_chosen = inputs['input_chosen']
                inputs_rejected = inputs['input_rejected']

                rewards_chosen, features_chosen = model.get_output_with_rep(
                    inputs_chosen
                )
                rewards_rejected, features_rejected = model.get_output_with_rep(
                    inputs_rejected
                )

            # calculate loss, optionally modulate with margin
            if "margin" in inputs:
                loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
            else:
                loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

            preferences = torch.sigmoid(rewards_chosen - rewards_rejected)
            

            all_rewards_chosen.extend(rewards_chosen.cpu().numpy())
            all_rewards_rejected.extend(rewards_rejected.cpu().numpy())
            all_features_chosen.extend(features_chosen.cpu().numpy())
            all_features_rejected.extend(features_rejected.cpu().numpy())
            all_preferences.extend(preferences.cpu().numpy())
            all_ids.extend(inputs['id'].cpu().numpy())
        if return_features:
            return loss, {
                "rewards_chosen": all_rewards_chosen,
                "rewards_rejected": all_rewards_rejected,
                "features_chosen": all_features_chosen,
                "features_rejected": all_features_rejected,
                "preferences": all_preferences,
                "id": all_ids
            }
        return loss
    
    def compute_uncertainties(self, runs, mode, all_preds):

        ensemble_df = []
        for run in runs:
             ensemble_df.append(all_preds[run.run_name][f'eval_{mode}'][['First', 'Second']].to_numpy())
             ids = all_preds[run.run_name][f'eval_{mode}'][['id']]
        print(f"Number of ensemble predictions loaded: {len(ensemble_df)}")
        
        uncertainties = compute_uncertanties(ensemble_df, ids)
        del ensemble_df
        return uncertainties
    

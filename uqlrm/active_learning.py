from configs import ActiveLearningConfig, RewardConfigWithSavedPredictions
from reward_modeling import build_reward_model, RewardTrainerWithCustomEval, RewardDataCollatorWithPaddingAndIndices
from dataset_utils import create_datasets, undersample_dataset, DataFrameDataset
from metrics import compute_uncertanties, compute_ensemble_accuracy
from parsing import ActiveLearningArguments
from utils import StopCallback
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from datasets import load_dataset
import os
from transformers.trainer_callback import CallbackHandler, TrainerState, TrainerControl, DefaultFlowCallback
from transformers.integrations import get_reporting_integration_callbacks
from transformers import HfArgumentParser

class ActiveLearningTrainer():
    r"""
    This Active Learning Trainer class implements a deep ensemble that actively requests labels
    based on the uncertainty of the predictions from unlabeled data.
    """
    def __init__(self,
                 script_args) -> None:
        self.script_args = script_args

        # Build Active Learning Config
        self.al_config = self._build_active_learning_config(script_args)

        self.base_model, self.tokenizer, self.peft_config = build_reward_model(self.script_args)

        train_dataset, eval_dataset, test_dataset, ood_dataset = create_datasets(script_args, self.tokenizer, ood=True)

        # Downsample training set to the pool size
        indices = list(range(len(train_dataset)))
        indices = shuffle(indices, random_state=script_args.seed)
        indices = indices[:self.al_config.pool_size]
        train_dataset = Subset(train_dataset, indices)
        
        self.df_train = pd.DataFrame([train_dataset[i] for i in range(len(train_dataset))])

        if script_args.undersample_eval:
            undersampled_eval = undersample_dataset(eval_dataset, script_args.undersample_ratio, script_args.seed)
            undersampled_test = undersample_dataset(test_dataset, script_args.undersample_ratio, script_args.seed)
            self.eval_sets = {"train": train_dataset, "eval": undersampled_eval, "test": undersampled_test, "ood": ood_dataset}
        else:
            self.eval_sets = {"train": train_dataset, "eval": eval_dataset, "test": test_dataset, "ood": ood_dataset}    

        self.batch = self.df_train[:self.al_config.initial_sample_size]
        self.batch.reset_index(drop=True, inplace=True)
        self.batch = DataFrameDataset(self.batch)
        self.df_train = self.df_train[self.al_config.initial_sample_size:]
        
        # compute num_epochs based on dataset length and hyperparameters
        self.num_epochs = 1 + (len(self.df_train) // self.al_config.active_batch_size)

        # Set callbacks for logging
        log_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks([self.script_args.log_with])
        self.callback_handler = CallbackHandler(
            log_callbacks, self.base_model, self.tokenizer, None, None
        )
        self.state = TrainerState(
            is_local_process_zero=True,
            is_world_process_zero=True,
        )

        self.control = TrainerControl()

        # Build Reward Modeling Configs
        self.runs = []
        for i in range(self.al_config.ensemble_size):
                self.runs.append(self._build_reward_config(script_args, f"{script_args.run_name}_{i}", self.num_epochs))

        self._check_parameters(self.al_config, self.runs)

    def _check_parameters(self, al_config, runs):
        # Assert if the proposed parameters match in the Active Learning Setup
        assert len(runs) == al_config.ensemble_size, "ensemble size must match number of reward models"
        assert len(runs) > 0, "you must have at least a single model in the ensemble"

        rm_config = runs[0]
        assert al_config.initial_sample_size % (rm_config.per_device_train_batch_size * rm_config.gradient_accumulation_steps) == 0, "initial sample size should match number of gradient updates"
        assert al_config.active_batch_size  % (rm_config.per_device_train_batch_size * rm_config.gradient_accumulation_steps) == 0, "active batch size should match samples for gradient update"

        assert rm_config.push_predictions_to_hub, "Push Predictions to Hub must be True for active learning setup"
        assert rm_config.save_predictions_steps == 1, "Save Prediction Steps must happen every evaluation loop (after each epoch)"

    # TODO: Implement option to extend dataset with new points and run more steps
    def train(self):
          seed = 0
          for epoch in range(self.num_epochs):
                # For each model, train separately in the sampled set:
                for run in self.runs:

                    if epoch == 0:
                        self.base_model, self.tokenizer, self.peft_config = build_reward_model(self.script_args)
                        # Needs to reinit the score layer because the cached version will always return the same weights for every ensemble member
                        # Also requires different seeds, otherwise it samples the same initial weights
                        self._reinit_linear_layer(self.base_model.score, self.script_args.score_init_std, seed)
                        seed += 1   


                    reward_collator = RewardDataCollatorWithPaddingAndIndices(self.tokenizer, max_length=run.max_length)

                    trainer = RewardTrainerWithCustomEval(
                        model=self.base_model,
                        tokenizer=self.tokenizer,
                        data_collator=reward_collator,
                        args=run,
                        train_dataset=self.batch,
                        eval_dataset=self.eval_sets,
                        peft_config=self.peft_config,
                    )

                    trainer.add_callback(StopCallback())

                    trainer.train(resume_from_checkpoint=(epoch != 0))

                    global_step = trainer.state.global_step      

                self.state.global_step = global_step
                # Generate Ensemble Predictions and Eval/Wandb
                for mode in self.eval_sets.keys():
                    if mode == "train":
                        acquisition_fn = self._eval_ensemble(mode, global_step, return_uncertainty=True)
                    else:
                        self._eval_ensemble(mode, global_step)

                # Select new batch points based on uncertainty
                nxt_batch_ids = self._select_next_batch_ids(acquisition_fn, self.al_config.heuristic, \
                                        self.al_config.active_batch_size, self.df_train, self.al_config.selection_strategy).to_frame()
                
                # Merge with current df and remove points from it
                self.batch = nxt_batch_ids.merge(self.df_train, on='id', how='inner')
                self.batch = DataFrameDataset(self.batch)

                # Remove these rows from initial dataset
                all_rows = self.df_train.merge(nxt_batch_ids, how='outer', on='id', indicator=True)
                self.df_train = all_rows[all_rows['_merge'] == 'left_only']
                self.df_train = self.df_train.drop(columns=['_merge'])

                del self.base_model, self.tokenizer
                # For each epoch, re-instatiate the base_model after deleting previous instance
                # The goal is to clean the previous computational graph and prevend headaches related to continuously loading new checkpoints
                self.base_model, self.tokenizer, self.peft_config = build_reward_model(self.script_args)
                
        

    def _build_active_learning_config(self, args) -> ActiveLearningConfig:
            return ActiveLearningConfig(
                    initial_sample_size=args.initial_sample_size,
                    ensemble_size=args.ensemble_size,
                    active_batch_size=args.active_batch_size,
                    pool_size=args.pool_size,
                    run_name=args.run_name,
                    heuristic=args.heuristic,
                    selection_strategy=args.selection_strategy,
                    output_dir=os.path.join(args.output_dir, "active_learning"))

    def _build_reward_config(self, args, run_name, num_epochs):
            return RewardConfigWithSavedPredictions(
                output_dir=os.path.join(args.output_dir, run_name),
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                num_train_epochs=num_epochs,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_checkpointing=args.gradient_checkpointing,
                learning_rate=args.learning_rate,
                report_to=args.log_with,
                remove_unused_columns=False,
                warmup_steps=args.num_warmup_steps,
                optim=args.optimizer_type,
                evaluation_strategy=args.eval_strategy,
                logging_strategy=args.logging_strategy,
                save_strategy=args.save_strategy,
                max_length=args.seq_length,
                run_name=run_name,
                push_predictions_to_hub=args.push_predictions_to_hub,
                predictions_dataset_hub=args.predictions_dataset_hub,
                save_predictions_steps=args.save_predictions_steps,
                save_total_limit=args.save_total_limit,
                bf16=args.bf16,
                log_level="debug")

    def _reinit_linear_layer(self, module, score_init_std, seed):
        module.weight.data.normal_(mean=0.0, std=score_init_std, generator=torch.manual_seed(seed))
        if module.bias is not None:
            module.bias.data.zero_()

    def _eval_ensemble(self, mode, global_step, return_uncertainty=False):
        ensemble_df = []
        for run in self.runs:
            datafile = os.path.join(self.script_args.output_dir, run.run_name, run.run_name, f"checkpoint-{global_step}", f"eval_{mode}", "predictions.csv")
            try: 
                df = load_dataset("luckeciano/uqlrm_predictions", data_files=datafile)['train'].to_pandas()
                ensemble_df.append(df)
            except:
                continue
        # TODO: Add ensemble args 
        print(f"Number of ensemble predictions loaded: {len(ensemble_df)}")
        
        epistemic, predictive, aleatoric, ens_probs, var_predictions, ids = compute_uncertanties(ensemble_df)
        avg_ep = epistemic.mean()
        avg_pred = predictive.mean()
        avg_ale = aleatoric.mean()
        avg_var = var_predictions.mean()
        acc = compute_ensemble_accuracy(ens_probs)
        log_likelihood = -log_loss(np.zeros(len(ens_probs['First'])), ens_probs.values, labels=[0, 1])
        logs = { 
            f"ensemble/{mode}_EnsAvgEpistemic": avg_ep, 
            f"ensemble{mode}_EnsAvgPredictive": avg_pred, 
            f"ensemble/{mode}_EnsAvgAleatoric": avg_ale, 
            f"ensemble/{mode}_EnsAvgVariance": avg_var, 
            f"ensemble/{mode}_EnsAvgAccuracy": acc,
            f"ensemble/{mode}_LogLikelihood": log_likelihood}
        
        self.callback_handler.on_log(self.al_config, self.state, self.control, logs)

        acquisition_fn = {}
        if return_uncertainty:
            acquisition_fn = {'Epistemic Uncertainty': epistemic, 'Predictive Uncertainty': predictive, 'Aleatoric Uncertainty': aleatoric, 'Variance': var_predictions, 'id': ids}

        return acquisition_fn
    
    def _select_next_batch_ids(self, acquisition_fn, heuristic, batch_size, current_pool, selection_strategy): 
        if heuristic == 'random':
            next_batch_ids = current_pool.sample(n = batch_size)
        else:
            df = acquisition_fn[heuristic]
            ids = acquisition_fn['id']
            df_id = pd.concat([df, ids], axis=1)
            final_pool = df_id.merge(current_pool, on='id', how='inner')

            if selection_strategy == 'rank':
                next_batch_ids = final_pool.nlargest(batch_size, heuristic)
            elif selection_strategy == 'sample':
                normalized_probs = final_pool[heuristic] / final_pool[heuristic].sum()
                next_batch_ids = final_pool.sample(n=batch_size, replace=False, weights=normalized_probs)
            
        return next_batch_ids['id']

parser = HfArgumentParser(ActiveLearningArguments)
script_args = parser.parse_args_into_dataclasses()[0]

trainer = ActiveLearningTrainer(script_args)
trainer.train()
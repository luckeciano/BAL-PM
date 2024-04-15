from dataset_utils import create_multiples_datasets
from parsing import ActiveLearningArguments
from utils import StopCallback, EvaluateAfterEpochCallback, push_predictions_to_hub
from factory import RewardModelFactory, DataCollatorFactory, TrainerFactory, DatasetFactory

import pandas as pd
import numpy as np
import torch
import os
from sklearn.utils import shuffle
from transformers.trainer_callback import CallbackHandler, TrainerState, TrainerControl, DefaultFlowCallback, EarlyStoppingCallback
from transformers.integrations import get_reporting_integration_callbacks
from transformers import HfArgumentParser
from active_learning import ActiveLearningTrainer
import ast

PANDAS_BATCH_SIZE = 2000

class ActiveLearningForPretrainingEnsemblingTrainer(ActiveLearningTrainer):
    r"""
    This Active Learning Trainer class implements a deep ensemble from different backbones that actively requests labels
    based on the uncertainty of the predictions from unlabeled data.
    """
    def __init__(self,
                 script_args) -> None:
        self.script_args = script_args

        # Build Active Learning Config
        self.al_config = self._build_active_learning_config(script_args)
        script_args.dataset_name = ast.literal_eval(script_args.dataset_name)

        assert len(script_args.dataset_name) == script_args.ensemble_size, "You must have one dataset per ensemble member."

        with open(self.script_args.clusters_filepath, 'r') as f:
            data = f.read()
            self.groups_dict = ast.literal_eval(data)

        self.base_model, self.tokenizer, self.peft_config = RewardModelFactory().create(self.script_args.model_type)(self.script_args)

        train_dataset, eval_dataset, test_dataset, ood_dataset =  create_multiples_datasets(script_args, self.tokenizer)

        # Downsample training set to the pool size
        indices = list(range(len(train_dataset[script_args.dataset_name[0]])))
        indices = shuffle(indices, random_state=script_args.seed)
        indices = indices[:self.al_config.pool_size]

        self.build_dataset = DatasetFactory().create(self.script_args.dataset_type)

        self.df_train = {}
        for dataset_name, dataset in train_dataset.items():
            full_train_df = dataset.to_pandas()
            self.df_train[dataset_name] = full_train_df.iloc[indices]
            train_dataset[dataset_name] = self.build_dataset(self.df_train[dataset_name])

        for dataset_name, dataset in eval_dataset.items():
            eval_dataset[dataset_name] = self.build_dataset(dataset.to_pandas())
        
        for dataset_name, dataset in test_dataset.items():
            test_dataset[dataset_name] = self.build_dataset(dataset.to_pandas())

        for dataset_name, dataset in ood_dataset.items():
            ood_dataset[dataset_name] = self.build_dataset(dataset.to_pandas())
        
        self.eval_sets = {}
        for dataset_name in script_args.dataset_name:
            self.eval_sets[dataset_name] = {"train": train_dataset[dataset_name], "eval": eval_dataset[dataset_name], "test": test_dataset[dataset_name], "ood": ood_dataset[dataset_name]}

        self.batch = {}
        for dataset_name, dataset in self.df_train.items():
            self.batch[dataset_name] = self.df_train[dataset_name][:self.al_config.initial_sample_size]
            self.batch[dataset_name].reset_index(drop=True, inplace=True)
            self.batch[dataset_name] = self.build_dataset(self.batch[dataset_name])
            self.df_train[dataset_name] = self.df_train[dataset_name][self.al_config.initial_sample_size:]
        
        # compute num_epochs based on dataset length and hyperparameters
        # self.num_epochs = 1 + (len(self.df_train) // self.al_config.active_batch_size)
        self.num_epochs = self.al_config.epoch_steps

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
        self.all_predictions = {}
        for i in range(self.al_config.ensemble_size):
                self.runs.append(self._build_reward_config(script_args, f"{script_args.run_name}_{i}", self.num_epochs))
                self.all_predictions[f"{script_args.run_name}_{i}"] = {}

        self._check_parameters(self.al_config, self.runs)

    def train(self):
        seed = 0
        for epoch in range(self.num_epochs):
            # For each model, train separately in the sampled set:
            for run, dataset_name in zip(self.runs, self.script_args.dataset_name):

                if epoch == 0 or self.al_config.training_strategy == "full_retrain":
                    # Input Size varies depending on the dataset
                    self.script_args.input_size = len(self.df_train[dataset_name].keys()) // 2
                    self.base_model, self.tokenizer, self.peft_config = RewardModelFactory().create(self.script_args.model_type)(self.script_args)
                    seed += 1   

                reward_collator = DataCollatorFactory().create(self.script_args.collator_type)(self.tokenizer, max_length=run.max_length)

                # Shuffle Dataset for new training
                self.batch[dataset_name].shuffle()

                trainer = TrainerFactory().create(self.script_args.trainer_type)(
                    model=self.base_model,
                    tokenizer=self.tokenizer,
                    collator=reward_collator,
                    run_args=run,
                    train_dataset=self.batch[dataset_name],
                    eval_datasets=self.eval_sets[dataset_name],
                    peft_config=self.peft_config,
                )


                if self.al_config.training_strategy == "full_retrain":
                    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
                    trainer.add_callback(EvaluateAfterEpochCallback())
                    trainer.train()
                    predictions = {}
                    
                    # Get preferences from buffer set
                    _, inference = trainer.inference(self.batch[dataset_name], return_features=True)
                    predictions["eval_buffer"] = self._build_inference_df(inference)
                    
                    # Get preferences for inference sets
                    for eval_dataset_name, eval_dataset in self.eval_sets[dataset_name].items():
                        _, inference = trainer.inference(eval_dataset, return_features=True)
                        predictions[f"eval_{eval_dataset_name}"] = self._build_inference_df(inference)
                else:
                    trainer.add_callback(StopCallback())
                    trainer.train(resume_from_checkpoint=(epoch != 0))
                    predictions = trainer.predictions
            
                self.all_predictions[run.run_name][epoch] = predictions
                self.run_dir = trainer.run_dir

            self.state.global_step = epoch
            # Generate Ensemble Predictions and Eval/Wandb
            for mode in ['train', 'test', 'eval', 'ood']:
                if mode == "train":
                    acquisition_fn = self._eval_uncertainty(mode, epoch, self.all_predictions, trainer, return_uncertainty=True)
                else:
                    self._eval_uncertainty(mode, epoch, self.all_predictions, trainer)

            # Eval ensemble for the current training buffer
            self._eval_uncertainty("buffer", epoch, self.all_predictions, trainer)

            # Select new batch points based on uncertainty
            current_pool = self.df_train[self.script_args.dataset_name[0]]
            nxt_batch_ids = self._select_next_batch_ids(acquisition_fn, self.al_config.heuristic, \
                                    self.al_config.active_batch_size, current_pool, self.al_config.selection_strategy).to_frame()
            
            # Merge with current df and remove points from it
            for dataset_name in self.batch.keys():
                new_batch = nxt_batch_ids.merge(self.df_train[dataset_name], on='id', how='inner')

                if self.al_config.dataset_strategy == 'full_labeled_set':
                    # Add new batch to buffer and shuffle rows
                    self.batch[dataset_name] = pd.concat([self.batch[dataset_name].df, new_batch]).sample(frac=1).reset_index(drop=True)
                    self.batch[dataset_name] = self.build_dataset(self.batch[dataset_name])
                elif self.al_config.dataset_strategy == 'batch_only':
                    self.batch[dataset_name] = self.build_dataset(new_batch)

                # Remove these rows from initial dataset
                all_rows = self.df_train[dataset_name].merge(nxt_batch_ids, how='outer', on='id', indicator=True)
                self.df_train[dataset_name] = all_rows[all_rows['_merge'] == 'left_only']
                self.df_train[dataset_name] = self.df_train[dataset_name].drop(columns=['_merge'])


            del self.base_model, self.tokenizer
            # For each epoch, re-instatiate the base_model after deleting previous instance
            # The goal is to clean the previous computational graph and prevend headaches related to continuously loading new checkpoints
            self.base_model, self.tokenizer, self.peft_config = RewardModelFactory().create(self.script_args.model_type)(self.script_args)
    
        # Upload Predictions to Hub
        if self.script_args.push_predictions_to_hub:
            full_dir = os.path.join(self.script_args.output_dir, "predictions")
            push_predictions_to_hub(full_dir, self.script_args.predictions_dataset_hub)

if __name__ == "__main__":
    parser = HfArgumentParser(ActiveLearningArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    trainer = ActiveLearningForPretrainingEnsemblingTrainer(script_args)
    trainer.train()
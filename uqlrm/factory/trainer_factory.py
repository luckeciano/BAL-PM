from reward_modeling import RewardTrainerWithCustomEval, AdapterEnsembleRewardTrainer, VariationalRewardTrainer, DDURewardTrainer, MCDropoutRewardTrainer

class TrainerFactory:
    def __init__(self):
        self.models = {
            'finetune_ensemble_trainer': self._create_reward_trainer,
            'adapters_ensemble_trainer': self._create_adapters_ensembles_trainer,
            'vatiational_trainer': self._create_variational_trainer,
            'ddu_trainer': self._create_ddu_trainer,
            'mc_dropout_trainer': self._create_mcdropout_trainer
        }

    def create(self, model_type):
        if model_type not in self.models:
            raise ValueError(f'Invalid model_type: {model_type}')
        return self.models[model_type]

    def _create_reward_trainer(self, model, tokenizer, collator, run_args, train_dataset, eval_datasets, peft_config):
        return RewardTrainerWithCustomEval(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    args=run_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_datasets,
                    peft_config=peft_config,
                )
    
    def _create_adapters_ensembles_trainer(self, model, tokenizer, collator, run_args, train_dataset, eval_datasets, peft_config):
        return AdapterEnsembleRewardTrainer(
            model=model,
            args=run_args,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets,
            data_collator=collator,
        )
    
    def _create_variational_trainer(self, model, tokenizer, collator, run_args, train_dataset, eval_datasets, peft_config):
        return VariationalRewardTrainer(
            model=model,
            args=run_args,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets,
            data_collator=collator
        )
    
    def _create_ddu_trainer(self, model, tokenizer, collator, run_args, train_dataset, eval_datasets, peft_config):
        return DDURewardTrainer(
            model=model,
            args=run_args,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets,
            data_collator=collator,
        )
    
    def _create_mcdropout_trainer(self, model, tokenizer, collator, run_args, train_dataset, eval_datasets, peft_config):
        return MCDropoutRewardTrainer(
            model=model,
            args=run_args,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets,
            data_collator=collator,
        )

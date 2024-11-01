import scipy.special
from configs import ActiveLearningConfig, RewardConfigWithSavedPredictions
from dataset_utils import create_datasets, create_features_dataset
from parsing import ActiveLearningArguments
from utils import StopCallback, EvaluateAfterEpochCallback, push_predictions_to_hub
from factory import RewardModelFactory, DataCollatorFactory, TrainerFactory, DatasetFactory
from metrics import compute_accuracy

import pandas as pd
import numpy as np
import torch
import gc
import tracemalloc
import os
import utils
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from transformers.trainer_callback import CallbackHandler, TrainerState, TrainerControl, DefaultFlowCallback, EarlyStoppingCallback
from transformers.integrations import get_reporting_integration_callbacks
from transformers import HfArgumentParser
import scipy
import ast

PANDAS_BATCH_SIZE = 2000

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

        if script_args.selection_strategy == 'clustered_rank':
            with open(self.script_args.clusters_filepath, 'r') as f:
                data = f.read()
                self.groups_dict = ast.literal_eval(data)

        if script_args.heuristic == "llm_uncertainty":
            self.llm_uncertainties = pd.read_csv(self.script_args.llm_unc_filepath)
        
        if script_args.log_batch_indices:
            import csv
            self.batch_idx_file = open(script_args.batch_idx_filepath, 'w', newline='')
            self.batch_idx_writer = csv.writer(self.batch_idx_file)

        self.base_model, self.tokenizer, self.peft_config = RewardModelFactory().create(self.script_args.model_type)(self.script_args)

        train_dataset, eval_dataset, test_dataset, ood_dataset = create_datasets(script_args, self.tokenizer)
        
        full_train_df = train_dataset.to_pandas()
        # Downsample training set to the pool size
        indices = list(range(len(train_dataset)))
        indices = shuffle(indices, random_state=script_args.seed)
        indices = indices[:self.al_config.pool_size]
        self.df_train = full_train_df.iloc[indices]

        self.build_dataset = DatasetFactory().create(self.script_args.dataset_type)

        train_dataset = self.build_dataset(self.df_train)
        eval_dataset = self.build_dataset(eval_dataset.to_pandas())
        test_dataset = self.build_dataset(test_dataset.to_pandas())
        ood_dataset = self.build_dataset(ood_dataset.to_pandas())

        if script_args.undersample_eval:
            undersampled_val_eval = self._undersample_dataset(eval_dataset, script_args.undersample_val_ratio, script_args.seed)
            
            undersampled_infer_eval = self._undersample_dataset(eval_dataset, script_args.undersample_infer_ratio, script_args.seed)
            undersampled_infer_test = self._undersample_dataset(test_dataset, script_args.undersample_infer_ratio, script_args.seed)
            self.eval_sets = {"eval": undersampled_val_eval}
            self.inference_sets = {"test": undersampled_infer_test, "eval": undersampled_infer_eval}
        else:
            self.eval_sets = {"train": train_dataset, "eval": eval_dataset, "test": test_dataset, "ood": ood_dataset}
            self.inference_sets = {"test": test_dataset, "eval": eval_dataset}

        # Adding OOD dataset to inference sets
        self.inference_sets['ood'] = ood_dataset  
        
        self.batch = self.df_train[:self.al_config.initial_sample_size]
        self.batch.reset_index(drop=True, inplace=True)
        self.batch = self.build_dataset(self.batch)
        self.df_train = self.df_train[self.al_config.initial_sample_size:]

        
        # Adding train set to inference sets based on downsample parameter
        self.inference_sets['train'] = self._downsample_pool(self.df_train, 16 * self.al_config.active_batch_size) \
            if self.al_config.downsample_pool else train_dataset
        
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

        if self.script_args.trainer_type == 'adapters_ensemble_trainer' or self.script_args.trainer_type == 'variational_trainer':
            assert not rm_config.gradient_checkpointing, "Gradient Checkpointing is not supported for Adapters/Variational Trainers"

        if self.script_args.dataset_strategy == 'full_labeled_set':
            assert rm_config.ignore_data_skip, "When using full labeled set, ignore Data Skip should be enabled, otherwise batches will be skipped"

    def train(self):
        seed = 0
        # tracemalloc.start() 
        for epoch in range(self.num_epochs):       
            self.train_loop(epoch, seed)
            
            # snapshot = tracemalloc.take_snapshot() 
            # top_stats = snapshot.statistics('lineno') 
            
            # for stat in top_stats[:10]: 
            #     print(stat)
    
        # Close Batch Ids Files
        if script_args.log_batch_indices:
            self.batch_idx_file.close()
        
        # Upload Predictions to Hub
        # if self.script_args.push_predictions_to_hub:
        #     full_dir = os.path.join(self.script_args.output_dir, "predictions")
        #     push_predictions_to_hub(full_dir, self.script_args.predictions_dataset_hub)
        
    def train_loop(self, epoch, seed):
        # For each model, train separately in the sampled set:
        all_predictions = {}
        for run in self.runs:

            if epoch == 0 or self.al_config.training_strategy == "full_retrain":
                self.base_model, self.tokenizer, self.peft_config = RewardModelFactory().create(self.script_args.model_type)(self.script_args)
                
                if script_args.model_type == "finetune_ens":
                    # Needs to reinit the score layer because the cached version will always return the same weights for every ensemble member
                    # Also requires different seeds, otherwise it samples the same initial weights
                    self._reinit_linear_layer(self.base_model.score, self.script_args.score_init_std, seed)
                seed += 1   


            reward_collator = DataCollatorFactory().create(self.script_args.collator_type)(self.tokenizer, max_length=run.max_length)

            # Shuffle Dataset for new training
            self.batch.shuffle()

            trainer = TrainerFactory().create(self.script_args.trainer_type)(
                model=self.base_model,
                tokenizer=self.tokenizer,
                collator=reward_collator,
                run_args=run,
                train_dataset=self.batch,
                eval_datasets=self.eval_sets,
                peft_config=self.peft_config,
            )


            if self.al_config.training_strategy == "full_retrain":
                trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=self.script_args.es_patience))
                trainer.add_callback(EvaluateAfterEpochCallback())
                trainer.train()
                predictions = {}
                
                # Get preferences from buffer set
                _, inference = trainer.inference(self.batch, return_features=True)
                predictions["eval_buffer"] = self._build_inference_df(inference)
                
                # Get preferences for inference sets
                for eval_dataset_name, eval_dataset in self.inference_sets.items():
                    _, inference = trainer.inference(eval_dataset, return_features=True)
                    predictions[f"eval_{eval_dataset_name}"] = self._build_inference_df(inference)
            else:
                trainer.add_callback(StopCallback())
                trainer.train(resume_from_checkpoint=(epoch != 0))
                predictions = trainer.predictions
        
            all_predictions[run.run_name] = predictions

        self.state.global_step = epoch
        # Generate Ensemble Predictions and Eval/Wandb
        
        for mode in self.inference_sets.keys():
            if mode == "train":
                acquisition_fn = self._eval_uncertainty(mode, epoch, all_predictions, trainer, return_uncertainty=True)
            else:
                self._eval_uncertainty(mode, epoch, all_predictions, trainer)

        # Eval ensemble for the current training buffer
        self._eval_uncertainty("buffer", epoch, all_predictions, trainer)

        # Select new batch points based on uncertainty
        nxt_batch_ids = self._select_next_batch_ids(epoch, acquisition_fn, self.al_config.heuristic, \
                                self.al_config.active_batch_size, self.df_train, self.al_config.selection_strategy).to_frame()
        
        # Log Batch Ids
        if script_args.log_batch_indices:
            batch_ids = np.array(nxt_batch_ids.values, dtype=int).flatten()
            self.batch_idx_writer.writerow(batch_ids)
        
        # Merge with current df and remove points from it
        new_batch = nxt_batch_ids.merge(self.df_train, on='id', how='inner')
        if self.al_config.dataset_strategy == 'full_labeled_set':
            # Add new batch to buffer and shuffle rows
            self.batch.extend(new_batch)
            del new_batch
        elif self.al_config.dataset_strategy == 'batch_only':
            self.batch = self.build_dataset(new_batch)

        # # Remove these rows from initial dataset
        #new_df_train = pd.merge(self.df_train, nxt_batch_ids, on='id', how='outer', indicator=True).query('_merge=="left_only"').drop(columns=['_merge'])
        self.df_train = self.df_train[~self.df_train['id'].isin(nxt_batch_ids['id'])]

        if self.al_config.downsample_pool:
            self.inference_sets['train'] = self._downsample_pool(self.df_train, 16 * self.al_config.active_batch_size)

        del nxt_batch_ids, inference, acquisition_fn, all_predictions, self.base_model, self.tokenizer, trainer
        torch.cuda.empty_cache()
        gc.collect()
        # For each epoch, re-instatiate the base_model after deleting previous instance
        # The goal is to clean the previous computational graph and prevend headaches related to continuously loading new checkpoints
        self.base_model, self.tokenizer, self.peft_config = RewardModelFactory().create(self.script_args.model_type)(self.script_args)


    def _build_active_learning_config(self, args) -> ActiveLearningConfig:
            return ActiveLearningConfig(
                    initial_sample_size=args.initial_sample_size,
                    ensemble_size=args.ensemble_size,
                    epoch_steps=args.epoch_steps,
                    active_batch_size=args.active_batch_size,
                    pool_size=args.pool_size,
                    downsample_pool=args.downsample_pool,
                    run_name=args.run_name,
                    heuristic=args.heuristic,
                    selection_strategy=args.selection_strategy,
                    dataset_strategy=args.dataset_strategy,
                    training_strategy=args.training_strategy,
                    gumbel_beta=args.gumbel_beta,
                    output_dir=os.path.join(args.output_dir, "active_learning"))

    def _build_reward_config(self, args, run_name, num_epochs):
            return RewardConfigWithSavedPredictions(
                output_dir=os.path.join(args.output_dir, run_name),
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                num_train_epochs=num_epochs if args.training_strategy != "full_retrain" else 4.0,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_checkpointing=args.gradient_checkpointing,
                learning_rate=args.learning_rate,
                report_to=args.log_with,
                remove_unused_columns=False,
                warmup_steps=args.num_warmup_steps,
                optim=args.optimizer_type,
                evaluation_strategy=args.evaluation_strategy,
                logging_strategy=args.logging_strategy,
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                save_strategy=args.save_strategy,
                max_length=args.seq_length,
                run_name=run_name,
                push_predictions_to_hub=args.push_predictions_to_hub,
                predictions_dataset_hub=args.predictions_dataset_hub,
                predictions_dir=os.path.join(args.output_dir, "predictions"),
                save_predictions_steps=args.save_predictions_steps,
                save_total_limit=args.save_total_limit,
                bf16=args.bf16,
                ignore_data_skip=args.ignore_data_skip,
                dataloader_num_workers=args.num_workers,
                dataloader_pin_memory=args.pin_memory,
                metric_for_best_model="eval_eval_loss",
                load_best_model_at_end=True,
                greater_is_better=False,
                regularized_loss=args.regularization_loss,
                lambda_regularization=args.lambda_regularizer,
                mc_dropout_realizations=args.mc_dropout_realizations,
                log_level="debug")

    def _reinit_linear_layer(self, module, score_init_std, seed):
        module.weight.data.normal_(mean=0.0, std=score_init_std, generator=torch.manual_seed(seed))
        if module.bias is not None:
            module.bias.data.zero_()

    def _eval_uncertainty(self, mode, epoch, all_preds, trainer, return_uncertainty=False):
        epistemic, predictive, aleatoric, ens_probs, var_predictions, ids = trainer.compute_uncertainties(self.runs, mode, all_preds)
        
        avg_ep = epistemic.mean()
        avg_pred = predictive.mean()
        avg_ale = aleatoric.mean()
        avg_var = var_predictions.mean()
        acc = compute_accuracy(ens_probs)
        model_error = (ens_probs['First'] < 0.5) * 1.0
        log_likelihood = -log_loss(np.zeros(len(ens_probs['First'])), ens_probs.values, labels=[0, 1])
        brier_score = (1 - ens_probs['First']) ** 2
        logs = { 
            f"ensemble/{mode}_EnsAvgEpistemic": avg_ep,
            f"ensemble{mode}_EnsAvgPredictive": avg_pred, 
            f"ensemble/{mode}_EnsAvgAleatoric": avg_ale, 
            f"ensemble/{mode}_EnsAvgVariance": avg_var, 
            f"ensemble/{mode}_EnsAvgAccuracy": acc,
            f"ensemble/{mode}_LogLikelihood": log_likelihood,
            f"ensemble/{mode}_AvgBrierScore": brier_score.mean(),
            f"ensemble/{mode}_CorrEpistemicBrierScore": epistemic.corr(brier_score, method='spearman'),
            f"ensemble/{mode}_CorrPredictiveBrierScore": predictive.corr(brier_score, method='spearman'),
            f"ensemble/{mode}_CorrVarianceBrierScore": var_predictions.corr(brier_score, method='spearman'),
            f"ensemble/{mode}_CorrEpistemicError": epistemic.corr(model_error, method='spearman'),
            f"ensemble/{mode}_CorrPredictiveError": predictive.corr(model_error, method='spearman'),
            f"ensemble/{mode}_CorrVarianceError": var_predictions.corr(model_error, method='spearman'),
            }
        
        self.callback_handler.on_log(self.al_config, self.state, self.control, logs)

        acquisition_fn = {}
        if return_uncertainty:
            acquisition_fn = {'Epistemic Uncertainty': epistemic, 'Predictive Uncertainty': predictive, 'Aleatoric Uncertainty': aleatoric, 'Variance': var_predictions, 'id': ids}

            if self.script_args.heuristic == "llm_uncertainty":
                ids_df = ids.to_frame()
                unc_df = self.llm_uncertainties[['id', self.script_args.llm_unc_type]]

                if self.script_args.llm_unc_low_perplexity_policy:
                    # Select points with low perplexity, instead of high perplexity
                    unc_df[self.script_args.llm_unc_type] = -unc_df[self.script_args.llm_unc_type]
                    
                acquisition_fn[script_args.heuristic] = ids_df.merge(unc_df, on='id', how='inner')[self.script_args.llm_unc_type].rename(script_args.heuristic)
                #TODO Add id column as index and return only heuristic
        
        return acquisition_fn
    
    def _select_next_batch_ids(self, epoch, acquisition_fn, heuristic, batch_size, current_pool, selection_strategy): 
        if heuristic == 'random':
            pool_size = len(current_pool)
            next_batch_ids = current_pool.sample(n = min(batch_size, pool_size))
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
            elif selection_strategy == 'clustered_rank':
                next_batch_ids = self._cluster_rank(heuristic, final_pool, batch_size)
            elif selection_strategy == "softmax_bald":
                assert self.al_config.gumbel_beta > 0, "Gumbel's beta must be greater than zero"
                scores_N = final_pool[heuristic]
                N = len(scores_N)
                final_pool['softmax_scores'] = self._get_softmax_samples(scores_N, self.al_config.gumbel_beta, N)
                next_batch_ids = final_pool.nlargest(batch_size, 'softmax_scores')
            elif selection_strategy == "power_bald":
                assert self.al_config.gumbel_beta > 0, "Gumbel's beta must be greater than zero"
                if self.script_args.gumbel_beta_annealing:
                    annealing_epochs = self.script_args.gumbel_beta_annealing_epochs
                    if epoch < self.script_args.gumbel_beta_annealing_start_epoch:
                        self.gumbel_beta = 0.25
                    elif epoch < annealing_epochs:
                        self.gumbel_beta += (self.al_config.gumbel_beta - 0.25) / annealing_epochs
                else:
                    self.gumbel_beta = self.al_config.gumbel_beta
                scores_N = final_pool[heuristic]
                N = len(scores_N)
                print(f"Gumbel Beta: {self.gumbel_beta}")
                final_pool['power_scores'] = self._get_power_samples(scores_N, self.gumbel_beta, N)
                next_batch_ids = final_pool.nlargest(batch_size, 'power_scores')
            elif selection_strategy == "softrank_bald":
                assert self.al_config.gumbel_beta > 0, "Gumbel's beta must be greater than zero"
                scores_N = final_pool[heuristic]
                N = len(scores_N)
                final_pool['softrank_scores'] = self._get_softrank_samples(scores_N, self.al_config.gumbel_beta, N)
                next_batch_ids = final_pool.nlargest(batch_size, 'softrank_scores')
            elif selection_strategy == "sample-then-rank":
                final_pool = final_pool.sample(frac=1).reset_index(drop=True)
                final_pool['batch'] = np.arange(len(final_pool)) // 16
                result = final_pool.groupby('batch').apply(lambda group: group.loc[group[heuristic].idxmax()]).sample(n=batch_size)
                next_batch_ids = result.reset_index(drop=True)
            elif selection_strategy == "state-entropy":
                device = next(self.base_model.parameters()).device
                state_entropies = self._compute_state_entropy(k=self.script_args.state_ent_k, device=device)
                final_pool = final_pool.merge(state_entropies, on='id', how='inner')
                final_pool['final_score'] = final_pool[heuristic] + self.script_args.state_ent_beta * final_pool['state_entropy']
                next_batch_ids = self._cluster_rank('final_score', final_pool, batch_size)
                #TODO Implement state entropy conditional to current data and then conditional on epistemic
            elif selection_strategy == "batch-state-entropy":
                device = next(self.base_model.parameters()).device
                pool = self._compute_batch_state_entropy(final_pool[['id', heuristic]], heuristic, self.batch, k=self.script_args.state_ent_k, device=device)
                next_batch_ids = pd.DataFrame(columns=['id']).astype({"id": int})
                
                if self.script_args.no_uncertainty:
                    pool[heuristic] = torch.zeros(pool[heuristic].shape).to(device)

                batch_stats = {}
                for _ in range(batch_size):
                    pool['ent_dist_term'] = torch.log(2.0 * pool['dists'] + 0.0001)
                    pool['ent_digamma_term'] = torch.digamma(pool['n_batch'] + 1.0)
                    pool['state_entropy'] = pool['ent_dist_term'] - (pool['ent_digamma_term'] / self.script_args.state_ent_d)
                    if self.script_args.normalize_entropy:
                        pool['state_entropy'] = (pool['state_entropy'] - pool['state_entropy'].mean()) / pool['state_entropy'].std()
                    
                    pool['entropy_score'] = self.script_args.state_ent_beta * pool['state_entropy']

                    if self.script_args.entropy_minimizer_baseline:
                        pool['final_score'] = pool[heuristic] - pool['entropy_score']
                    else:
                        pool['final_score'] = pool[heuristic] + pool['entropy_score']
                    
                    pool['uncertainty_score_ratio'] = pool[heuristic] / pool['final_score']
                    
                    max_score_index = torch.argmax(pool['final_score'])
                    # Create a mask for all rows except the one with the maximum final_score
                    mask = torch.ones(len(pool['final_score']), dtype=torch.bool)
                    mask[max_score_index] = 0

                    # Select the row with the maximum final_score
                    row = {k: v[max_score_index].unsqueeze(0) for k, v in pool.items()}

                    for k, v in row.items():
                        if k not in ['ent_dist_term', 'ent_digamma_term', 'state_entropy', 'entropy_score' , 'final_score', 'uncertainty_score_ratio', heuristic]:
                            continue
                        if k not in batch_stats:
                            batch_stats[k] = [v]
                        else:
                            batch_stats[k].append(v)

                    # Drop the row from the pool
                    pool = {k: v[mask] for k, v in pool.items()}
                    
                    # row = pool[pool['final_score'] == pool['final_score'].max()]
                    row_df = pd.DataFrame({'id': row['id'].cpu().numpy()})
                    next_batch_ids = pd.concat([next_batch_ids, row_df], ignore_index=True)

                    # Update n batches
                    fts = pool['features'] # or drop other columns...
                    new_point = row['features']
                    n_batch = utils.compute_nbatch(pool['dists'], fts, new_point, device)
                    pool['n_batch'] =  pool['n_batch'] + n_batch

                logs = {}
                for k, v in batch_stats.items():
                    v_cat = torch.cat(v)
                    logs[f"batch_stats/{k}_max"] = torch.max(v_cat).cpu().numpy()
                    logs[f"batch_stats/{k}_min"] = torch.min(v_cat).cpu().numpy()
                    logs[f"batch_stats/{k}_p95"] = torch.quantile(v_cat, 0.95).cpu().numpy()
                    logs[f"batch_stats/{k}_p5"] = torch.quantile(v_cat, 0.05).cpu().numpy()
                    logs[f"batch_stats/{k}_p50"] = torch.quantile(v_cat, 0.50).cpu().numpy()
                
                self.callback_handler.on_log(self.al_config, self.state, self.control, logs)

            elif selection_strategy == "batch-state-entropy-v2":
                device = next(self.base_model.parameters()).device
                next_batch_ids = self._compute_batch_state_entropy_v2(final_pool[['id', heuristic]], heuristic, self.batch, ent_k=self.script_args.state_ent_k, device=device, batch_size=batch_size)
                
            del final_pool
                    
                
        return next_batch_ids['id']
    
    def _cluster_rank(self, heuristic, final_pool, batch_size):
        #TODO improve implementation
        num_collisions = 0
        final_pool.sort_values(heuristic, ascending=False, inplace=True)
        chosen_groups = set()
        next_batch_ids = pd.DataFrame(columns=final_pool.columns)
        for _, row in final_pool.iterrows():
            group = self.groups_dict[row['id']]
            if group not in chosen_groups:
                chosen_groups.add(group)
                next_batch_ids = pd.concat([next_batch_ids, pd.DataFrame([row])], ignore_index=True)
            else:
                num_collisions += 1
            
            if len(next_batch_ids) == batch_size:
                print(f"Num collisions: {num_collisions}")
                break
        
        return next_batch_ids
    
    def _get_softmax_samples(self, scores_N, beta, N):
        return scores_N + scipy.stats.gumbel_r.rvs(loc=0, scale=1 / beta, size=N, random_state=None)
    
    def _get_power_samples(self, scores_N, beta, N):
        return self._get_softmax_samples(np.log(scores_N + 0.00001), beta, N)

    def _get_softrank_samples(self, scores_N, beta, N):
        sorted_indices = np.argsort(-scores_N)
        ranks_N = np.argsort(sorted_indices) + 1
        return self._get_power_samples(1 / ranks_N, beta, N)
    
    def _compute_state_entropy(self, k, device):
        if getattr(self, "state_features", None) is None:
            self.state_features = create_features_dataset(self.script_args.state_features_dataset_name)
        fts = self.state_features.drop(columns='id')
        fts_pt = torch.Tensor(fts.values).to(device)
        ids_entropies = self.state_features['id'].astype(int).to_frame()
        dists = utils.find_kth_nearest_dists(fts_pt, k, device=device).cpu().numpy()
        state_entropies = np.log(2.0 * dists + 0.0001)
        norm_state_entropies = (state_entropies - np.mean(state_entropies)) / np.std(state_entropies)
        ids_entropies['state_entropy'] = norm_state_entropies
        ids_entropies = pd.concat([ids_entropies, fts], axis=1)
        return ids_entropies
    
    def _compute_batch_state_entropy(self, final_pool, heuristic, batch, k, device):
        if getattr(self, "state_features", None) is None:
            self.state_features = create_features_dataset(self.script_args.state_features_dataset_name)

        pool = final_pool.merge(self.state_features, on='id', how='inner')
        batch_fts = batch.get_df()[['id']].merge(self.state_features, on='id', how='inner').drop(columns='id').values
        batch_fts_pt = torch.Tensor(batch_fts).to(device)
        fts = pool.drop(columns=final_pool.columns)
        
        if self.script_args.normalize_state_features:
            mean = fts.mean()
            std = fts.std()
            fts = (fts - mean / std)
        
        fts_pt = torch.Tensor(fts.values).to(device)
        ids = torch.Tensor(pool['id']).to(device, dtype=torch.int64)
        uncertainties = torch.Tensor(pool[heuristic]).to(device)
        dists = utils.find_kth_nearest_dists(fts_pt, k, device=device)
        n_batch = utils.compute_nbatch(dists, fts_pt, batch_fts_pt, device)
        res = {'id': ids, heuristic: uncertainties, 'dists': dists, 'n_batch': n_batch, 'features': fts_pt}
        return res
    
    def _compute_batch_state_entropy_v2(self, final_pool, heuristic, batch, ent_k, device, batch_size):
        if getattr(self, "state_features", None) is None:
            self.state_features = create_features_dataset(self.script_args.state_features_dataset_name)

        pool = final_pool.merge(self.state_features, on='id', how='inner')
        batch_fts = batch.get_df()[['id']].merge(self.state_features, on='id', how='inner').drop(columns='id').values
        batch_fts_pt = torch.Tensor(batch_fts).to(device, dtype=torch.float64)
        fts = pool.drop(columns=final_pool.columns)
        
        if self.script_args.normalize_state_features:
            mean = fts.mean()
            std = fts.std()
            fts = (fts - mean / std)
        
        fts_pt = torch.Tensor(fts.values).to(device, dtype=torch.float64)
        ids = torch.Tensor(pool['id']).to(device, dtype=torch.int64)
        uncertainties = torch.Tensor(pool[heuristic]).to(device)

        dists = utils.compute_batch_dists(fts_pt, batch_fts_pt, device=device) # (N_pool + N_batch, N_pool + N_batch)

        pool = {'id': ids, heuristic: uncertainties, 'dists': dists, 'pool_fts': fts_pt} 

        next_batch_ids = pd.DataFrame(columns=['id']).astype({"id": int})
                
        if self.script_args.no_uncertainty:
            pool[heuristic] = torch.zeros(pool[heuristic].shape).to(device)

        batch_stats = {}
        with torch.no_grad():
            for _ in range(batch_size):
                torch.cuda.empty_cache()
                gc.collect()

                # compute knns
                topk_dists, _ = torch.topk(pool['dists'], ent_k, largest=False) # (N_pool, )

                pool['state_entropy'] = torch.log(2.0 * topk_dists[:, ent_k-1] + 0.0001)
                if self.script_args.normalize_entropy:
                    pool['state_entropy'] = (pool['state_entropy'] - pool['state_entropy'].mean()) / pool['state_entropy'].std()
                
                pool['entropy_score'] = self.script_args.state_ent_beta * pool['state_entropy']
                pool['final_score'] = pool[heuristic] + pool['entropy_score']
                pool['uncertainty_score_ratio'] = pool[heuristic] / pool['final_score']
                
                max_score_index = torch.argmax(pool['final_score'])
                # Create a mask for all rows except the one with the maximum final_score
                mask = torch.ones(len(pool['final_score']), dtype=torch.bool)
                mask[max_score_index] = 0

                # Select the row with the maximum final_score
                row = {k: v[max_score_index].unsqueeze(0) for k, v in pool.items()}

                for k, v in row.items():
                    if k not in ['state_entropy', 'entropy_score' , 'final_score', 'uncertainty_score_ratio', heuristic]:
                        continue
                    if k not in batch_stats:
                        batch_stats[k] = [v]
                    else:
                        batch_stats[k].append(v)

                del pool['final_score'], pool['uncertainty_score_ratio'], pool['state_entropy']

                # drop row from pool
                old_pool = pool
                pool = {k: v[mask] for k, v in old_pool.items()}
                del old_pool
                # compute dists for new pool and row
                acquired_pt_dists = utils.compute_batch_dists(pool['pool_fts'], row['pool_fts'], device=device) # (N_pool, 1)

                # add column to dists
                old_pool_dists = pool['dists']
                pool['dists'] = torch.cat((old_pool_dists, acquired_pt_dists), dim=1)
                
                
                row_df = pd.DataFrame({'id': row['id'].cpu().numpy()})
                next_batch_ids = pd.concat([next_batch_ids, row_df], ignore_index=True)
                
                del row, old_pool_dists
                

        logs = {}
        for k, v in batch_stats.items():
            v_cat = torch.cat(v).float()
            logs[f"batch_stats/{k}_max"] = torch.max(v_cat).cpu().numpy()
            logs[f"batch_stats/{k}_min"] = torch.min(v_cat).cpu().numpy()
            logs[f"batch_stats/{k}_p95"] = torch.quantile(v_cat, 0.95).cpu().numpy()
            logs[f"batch_stats/{k}_p5"] = torch.quantile(v_cat, 0.05).cpu().numpy()
            logs[f"batch_stats/{k}_p50"] = torch.quantile(v_cat, 0.50).cpu().numpy()
        
        self.callback_handler.on_log(self.al_config, self.state, self.control, logs)

        return next_batch_ids

    
    def _undersample_dataset(self, dataset, ratio, seed):
        indices = list(range(len(dataset)))
        size = int(ratio * len(indices))
        indices = shuffle(indices, random_state=seed)
        indices = indices[:size]
        return self.build_dataset(dataset.get_df().iloc[indices])

    def _downsample_pool(self, df_train, new_size):
        indices = list(range(len(df_train)))
        indices = shuffle(indices)
        indices = indices[:new_size]
        return self.build_dataset(df_train.iloc[indices])

    def _build_inference_df(self, inference):
        if self.script_args.model_type == "mc_dropout":
            realizations = []
            for realization in inference:
                df = self._build_single_inference_df(realization)
                realizations.append(df)
            return realizations
        else:
            return self._build_single_inference_df(inference)
           
    def _build_single_inference_df(self, inference):
        preferences_np = np.array(inference['preferences']).flatten()
        rw_chosen_np = np.array(inference['rewards_chosen']).flatten() if 'rewards_chosen' in inference else None
        rw_rejected_np = np.array(inference['rewards_rejected']).flatten() if 'rewards_rejected' in inference else None
        ids = inference['id']
        return pd.DataFrame({"First": preferences_np, "Second": 1 - preferences_np, 'id': ids, 
                        'reward_chosen': rw_chosen_np, 'reward_rejected': rw_rejected_np})

if __name__ == "__main__":
    gc.enable()
    # gc.set_debug(gc.DEBUG_LEAK)
    parser = HfArgumentParser(ActiveLearningArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    trainer = ActiveLearningTrainer(script_args)
    trainer.train()
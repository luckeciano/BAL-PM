from datasets import load_dataset, Dataset
from dataset_utils import dataset_process_factory
import numpy as np
import pandas as pd

def process_and_filter_dataset(script_args, dataset, tokenizer):
    final_dataset = dataset.map(lambda example:
        getattr(dataset_process_factory, script_args.preprocess_fn)(example, tokenizer, script_args.seq_length),
        batched=True,
        # num_proc=4,
    )
    final_dataset = final_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
        and len(x["input_ids_rejected"]) <= script_args.seq_length
    )
    print(f"Size of the set before processing: {len(dataset)}, after processing: {len(final_dataset)}")
    return final_dataset

def create_multiples_datasets(args, tokenizer=None):
    train_set, valid_set, test_set, ood_set = {}, {}, {}, {}
    assert isinstance(args.dataset_name, list), "To create multiples dataset, you must pass a list of names."
    for dataset_name in args.dataset_name:
        train, valid, test, ood = create_single_dataset(args, dataset_name, tokenizer)

        train_set[dataset_name] = train
        valid_set[dataset_name] = valid
        test_set[dataset_name] = test
        ood_set[dataset_name] = ood

    return train_set, valid_set, test_set, ood_set

def create_datasets(args, tokenizer=None):
    return create_single_dataset(args, args.dataset_name, tokenizer)

def create_single_dataset(args, dataset_name, tokenizer=None):
    dataset = load_dataset(
        dataset_name,
        #use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None
    )

    train_dataset = dataset[args.train_split] if args.train_split in dataset else None
    valid_dataset = dataset[args.valid_split] if args.valid_split in dataset else None
    test_dataset = dataset[args.test_split] if args.test_split in dataset else None
    ood_dataset = dataset[args.ood_split] if args.ood_split in dataset else None

    if tokenizer is not None:
        final_train_dataset = process_and_filter_dataset(args, train_dataset, tokenizer)
        final_valid_dataset = process_and_filter_dataset(args, valid_dataset, tokenizer)
        final_test_dataset = process_and_filter_dataset(args, test_dataset, tokenizer)
        final_ood_dataset = process_and_filter_dataset(args, ood_dataset, tokenizer)

        return final_train_dataset, final_valid_dataset, final_test_dataset, final_ood_dataset

    return train_dataset, valid_dataset, test_dataset, ood_dataset

def create_features_dataset(dataset_name):
    return load_dataset(dataset_name)['train'].to_pandas()

def undersample_dataset(dataset, ratio, seed):
    dataset = dataset.train_test_split(test_size=ratio, seed=seed)
    return dataset["test"]

class DataFrameDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        batch =  self.df.iloc[idx]
        batch.reset_index(inplace=True, drop=True)
        return batch
    
    def shuffle(self):
        shuffled_df = self.df.sample(frac=1).reset_index(drop=True)
        self.df = shuffled_df
    
    def get_df(self):
        return self.df
    
    def extend(self, new_batch_df):
        # Check if all columns in self.df.columns are also present in new_batch_df
        if not set(self.df.columns).issubset(new_batch_df.columns):
            # If not, raise an error
            raise ValueError("new_batch_df must have at least all the columns in the DataFrameDataset")

        # Rearrange the columns of new_batch_df to match those in the DataFrameDataset
        new_batch_df = new_batch_df[self.df.columns]

        # Concatenate new_batch_df with the existing DataFrame
        self.df = pd.concat([self.df, new_batch_df], ignore_index=True)
    
class NumPyDataset(Dataset):
    def __init__(self, df):
        self.values = df.to_numpy()
        self.columns = df.columns

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        batch =  self.values[idx]
        return batch
    
    def __getitems__(self, idx):
        batch =  self.values[idx]
        return batch
    
    def shuffle(self):
        np.random.shuffle(self.values)
    
    def get_df(self):
        return pd.DataFrame(self.values, columns=self.columns)
    
    def extend(self, new_batch_df):
        # Check if all columns in self.columns are also present in new_batch_df
        if not set(self.columns).issubset(new_batch_df.columns):
            # If not, raise an error
            raise ValueError("new_batch_df must have at least all the columns in the NumPyDataset")

        # Rearrange the columns of new_batch_df to match those in the NumPyDataset
        new_batch_df = new_batch_df[self.columns]

        # Convert new_batch_df to a NumPy array and concatenate it with the existing values
        new_values = new_batch_df.to_numpy()
        self.values = np.concatenate((self.values, new_values), axis=0)
from datasets import load_dataset, Dataset
from dataset_utils import dataset_process_factory

def process_and_filter_dataset(script_args, dataset, tokenizer):
    final_dataset = dataset.map(lambda example:
        getattr(dataset_process_factory, script_args.preprocess_fn)(example, tokenizer, script_args.seq_length),
        batched=True,
        num_proc=4,
    )
    final_dataset = final_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
        and len(x["input_ids_rejected"]) <= script_args.seq_length
    )
    print(f"Size of the set before processing: {len(dataset)}, after processing: {len(final_dataset)}")
    return final_dataset

def create_datasets(args, tokenizer=None):
    dataset = load_dataset(
        args.dataset_name,
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
    
class NumPyDataset(Dataset):
    def __init__(self, df):
        self.values = df.values
        self.x = 0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        batch =  self.values[idx]
        return batch
    
    def __getitems__(self, idx):
        batch =  self.values[idx]
        return batch
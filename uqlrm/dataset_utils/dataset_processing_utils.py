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

def create_datasets(args, tokenizer, ood=False):
    dataset = load_dataset(
        args.dataset_name,
        #use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None
    )

    train_dataset = dataset[args.train_split]
    valid_dataset = dataset[args.valid_split]
    test_dataset = dataset[args.test_split]

    final_train_dataset = process_and_filter_dataset(args, train_dataset, tokenizer)
    final_valid_dataset = process_and_filter_dataset(args, valid_dataset, tokenizer)
    final_test_dataset = process_and_filter_dataset(args, test_dataset, tokenizer)

    if ood:
        ood_dataset = dataset[args.ood_split]
        final_ood_dataset = process_and_filter_dataset(args, ood_dataset, tokenizer)
        return final_train_dataset, final_valid_dataset, final_test_dataset, final_ood_dataset

    return final_train_dataset, final_valid_dataset, final_test_dataset

def undersample_dataset(dataset, ratio):
    dataset = dataset.train_test_split(test_size=ratio, seed=42)
    return dataset["test"]

class DataFrameDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import HfArgumentParser
from parsing import RewardModelingArguments
from torch.utils.data import DataLoader
from dataset_utils import create_datasets
from collators import RewardDataCollatorWithPaddingAndIndices
import numpy as np
from tqdm import tqdm
import csv

def main(script_args):
    # Initialize the tokenizer and model
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, trust_remote_code=True, truncation=True, max_length=script_args.seq_length)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    model.config.pad_token_id = model.config.eos_token_id # fix
    

    train_dataset, _, _, _ = create_datasets(script_args, tokenizer)

    # Check if a GPU is available and if not, use a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)

    # Set the model in evaluation mode
    model.eval()

    # Create a DataLoader to handle batching of the dataset
    dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=RewardDataCollatorWithPaddingAndIndices(tokenizer, max_length=script_args.seq_length))

    with open('./entropies_reddit_hermes.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['id', 'avg_entropy', 'max_entropy', 'entropy_interval'])
    
        # Iterate over each batch in the dataset
        for batch in tqdm(dataloader):
            # Get the input_ids and attention_mask
            input_ids = torch.tensor(batch['input_ids_chosen']).to(device)
            attention_mask = torch.tensor(batch['attention_mask_chosen']).to(device)


            # Calculate the softmax probabilities for the tokens
            logits = model(input_ids, attention_mask=attention_mask).logits
            token_probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Calculate the entropy of the softmax probabilities
            entropy = -np.sum((token_probs.detach().cpu().numpy() * np.log2(token_probs.detach().cpu().numpy())), axis=-1)

            # Calculate the average, max, and min entropies for each sentence, ignoring masked tokens
            for i in range(entropy.shape[0]):
                sentence_entropy = entropy[i][attention_mask[i].bool().cpu().numpy()]
                avg_entropy = np.mean(sentence_entropy)
                max_entropy = np.max(sentence_entropy)
                min_entropy = np.min(sentence_entropy)
                entropy_interval = max_entropy - min_entropy
            
            # Write the data to the CSV file
                writer.writerow([batch['id'][i], avg_entropy, max_entropy, entropy_interval])

if __name__ == "__main__":
    parser = HfArgumentParser(RewardModelingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)
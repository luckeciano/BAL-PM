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

def get_entropies(batch, model, input_name, mask_name, device):
    # Get the input_ids and attention_mask
    input_ids = torch.tensor(batch[input_name]).to(device)
    attention_mask = torch.tensor(batch[mask_name]).to(device)


    # Calculate the softmax probabilities for the tokens
    logits = model(input_ids, attention_mask=attention_mask).logits
    token_probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Calculate the entropy of the softmax probabilities
    entropy = -np.sum((token_probs.detach().cpu().numpy() * np.log2(token_probs.detach().cpu().numpy())), axis=-1)

    return entropy, attention_mask


def compute_entropy_stats(sentence_entropies):
    avg_entropy = np.mean(sentence_entropies)
    sum_entropy = np.sum(sentence_entropies)
    max_entropy = np.max(sentence_entropies)
    max_entropy = np.max(sentence_entropies)
    min_entropy = np.min(sentence_entropies)
    entropy_interval = max_entropy - min_entropy

    return avg_entropy, sum_entropy, max_entropy, entropy_interval


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
        writer.writerow(['id', 'avg_entropy_chosen', 'sum_entropy_chosen', 'max_entropy_chosen', 'entropy_interval_chosen', \
                         'avg_entropy_rejected', 'sum_entropy_rejected', 'max_entropy_rejected', 'entropy_interval_rejected', \
                            'total_avg_entropy', 'total_max_entropy', 'total_sum_entropy'])
    
        # Iterate over each batch in the dataset
        for batch in tqdm(dataloader):
            entropy_chosen, attn_mask_chosen = get_entropies(batch, model, 'input_ids_chosen', 'attention_mask_chosen', device)
            entropy_rejected, attn_mask_rejected = get_entropies(batch, model, 'input_ids_rejected', 'attention_mask_rejected', device)
            

            # Calculate the average, max, and min entropies for each sentence, ignoring masked tokens
            for i in range(entropy_chosen.shape[0]):
                sentence_entropy_chosen = entropy_chosen[i][attn_mask_chosen[i].bool().cpu().numpy()]
                sentence_entropy_rejected = entropy_rejected[i][attn_mask_rejected[i].bool().cpu().numpy()]

                avg_entropy_c, sum_entropy_c, max_entropy_c, entropy_interval_c = compute_entropy_stats(sentence_entropy_chosen)
                avg_entropy_r, sum_entropy_r, max_entropy_r, entropy_interval_r = compute_entropy_stats(sentence_entropy_rejected)
                total_sum_ent = sum_entropy_c + sum_entropy_r
                total_avg_ent = total_sum_ent / (len(sentence_entropy_chosen) + len(sentence_entropy_rejected))
                total_max = np.max([max_entropy_c, max_entropy_r])
            
            # Write the data to the CSV file
                writer.writerow([batch['id'][i], avg_entropy_c, sum_entropy_c, max_entropy_c, entropy_interval_c, avg_entropy_r, sum_entropy_r, max_entropy_r, entropy_interval_r, total_avg_ent, total_max, total_sum_ent])

if __name__ == "__main__":
    parser = HfArgumentParser(RewardModelingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)
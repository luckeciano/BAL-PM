from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def evaluate_model(model, tokenizer, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    for batch in tqdm(dataloader):
        inputs = tokenizer(batch[0], return_tensors='pt')
        labels = tokenizer(batch[1], return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :]
        predictions = torch.argmax(next_token_logits, dim=-1)
        labels = labels['input_ids'][:, 0]
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += predictions.numel()

    accuracy = correct_predictions / total_predictions
    return accuracy


class LAMBADADataset(Dataset):
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]


# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the LAMBADA dataset using Hugging Face's datasets library
lambada_dataset = load_dataset('lambada', split='test')

# Create a DataLoader for the LAMBADA dataset
text = lambada_dataset['text']
ctxs = []
labels = []
for t in text:
   seps = t.rsplit(" ", 1)
   ctxs.append(seps[0])
   labels.append(" " + seps[1]) 

dataloader = DataLoader(LAMBADADataset(ctxs, labels))

# Evaluate the model on the LAMBADA dataset
accuracy = evaluate_model(model, tokenizer, dataloader)
print(f"Accuracy on LAMBADA dataset: {accuracy}")


# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets

def chosen_rejected_preprocess_function(examples, tokenizer, max_len):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "id": []
    }
    for chosen, rejected, id in zip(examples["chosen"], examples["rejected"], examples["id"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["id"].append(id)

    return new_examples


def redditcnn_preprocess_function(examples, tokenizer, max_len):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "id": []
    }
    for post, chosen, rejected, id in zip(examples['post'], examples["chosen_summary"], examples["rejected_summary"], examples["id"]):
        final_post = f"Post: {post}"
        final_chosen = f"\nSummary: {chosen}"
        final_rejected = f"\nSummary: {rejected}"
        tokenized_chosen = tokenizer(final_chosen)
        tokenized_rejected = tokenizer(final_rejected)

        tokens_left = max_len - max(len(tokenized_chosen['input_ids']), len(tokenized_rejected['input_ids']))
        tokenized_post = tokenizer(final_post)
        tokenized_post['input_ids'] = tokenized_post['input_ids'][:(tokens_left // 2)]
        tokenized_post['attention_mask'] = tokenized_post['attention_mask'][:(tokens_left // 2)]

        final_tok_chosen = {}
        final_tok_chosen['input_ids'] = tokenized_post['input_ids'] + tokenized_chosen['input_ids']
        final_tok_chosen['attention_mask'] = tokenized_post['attention_mask'] + tokenized_chosen['attention_mask']

        final_tok_rejected = {}
        final_tok_rejected['input_ids'] = tokenized_post['input_ids'] + tokenized_rejected['input_ids']
        final_tok_rejected['attention_mask'] = tokenized_post['attention_mask'] + tokenized_rejected['attention_mask']

        new_examples["input_ids_chosen"].append(final_tok_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(final_tok_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(final_tok_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(final_tok_rejected["attention_mask"])
        new_examples["id"].append(id)

    return new_examples
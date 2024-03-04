# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
from sklearn.utils import shuffle
from typing import Literal
import json
import ast


def process_reddit_sft(example, tokenizer, seq_len):
    final_examples = {
        "text": []
    }
    
    for post, chosen in zip(example['post'], example["chosen_summary"]):
        final_post = f"Post: {post}"
        final_chosen = f"\nSummary: {chosen}"
        final_examples["text"].append(final_post + final_chosen)

    return final_examples

def process_alpacafarm_sft(example, tokenizer, seq_len):
    final_examples = {
        "text": []
    }
    
    for instruction, input, output in zip(example['instruction'], example["input"], example['output']):
        final_prompt = f"Instruction: {instruction}"

        response = f"\nResponse: {output}"

        if example['input']:
            final_prompt = final_prompt + f"\n{input}"
        
        final_prompt = final_prompt + response
        final_examples["text"].append(final_prompt)

    return final_examples

def process_ultrafeedback_sft(examples, tokenizer, max_len):
    new_examples = {
        "text": []
    }

    for post, chosen, rejected, messages,  id in zip(examples['prompt'], examples["chosen"], examples["rejected"], examples["messages"], examples["id"]):
        messages_dict = ast.literal_eval(messages.replace("}\n", "},"))
        example = {'post': post, 'chosen': chosen, 'rejected': rejected, 'messages': messages_dict}
        
        example = apply_chat_template(example, tokenizer, "sft")

        new_examples["text"].append(example['text'])
        
    return new_examples

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

def shuffle_tokens(examples, tokenizer, max_len):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "id": []
    }

    for input_chosen, mask_chosen, input_rejected, mask_rejected, id in zip(examples["input_ids_chosen"], \
            examples["attention_mask_chosen"], examples["input_ids_rejected"], examples["attention_mask_rejected"], examples["id"]):
        shuffled_input_chosen, shuffled_mask_chosen = shuffle(input_chosen, mask_chosen)
        shuffled_input_rejected, shuffled_mask_rejected = shuffle(input_rejected, mask_rejected)

        new_examples["input_ids_chosen"].append(shuffled_input_chosen)
        new_examples["attention_mask_chosen"].append(shuffled_mask_chosen)
        new_examples["input_ids_rejected"].append(shuffled_input_rejected)
        new_examples["attention_mask_rejected"].append(shuffled_mask_rejected)
        new_examples["id"].append(id)

    return new_examples
        

def ultrafeedback_preprocess_function(examples, tokenizer, max_len):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "id": []
    }

    for post, chosen, rejected, id in zip(examples['prompt'], examples["chosen"], examples["rejected"], examples["id"]):
        chosen = ast.literal_eval(chosen.replace("}\n", "},"))
        rejected = ast.literal_eval(rejected.replace("}\n", "},"))

        example = {"chosen": chosen, "rejected": rejected}
        example = apply_chat_template(example, tokenizer, "rm")

        final_chosen = example['text_chosen']
        final_rejected = example['text_rejected']
        tokenized_chosen = tokenizer(final_chosen)
        tokenized_rejected = tokenizer(final_rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["id"].append(id)

    return new_examples

def alpacafarm_preprocess_function(examples, tokenizer, max_len):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "id": []
    }

    for instruction, input, output_1, output_2, preference, id in zip(examples['instruction'], examples["input"], examples["output_1"], examples["output_2"], examples["preference"], examples["id"]):
        chosen = output_1 if preference == 1 else output_2
        rejected = output_2 if preference == 1 else output_1

        final_prompt = f"Instruction: {instruction}"
        if input:
            final_prompt = final_prompt + f"\n{input}"
        
        chosen_response = final_prompt + f"\nResponse: {chosen}"
        rejected_response = final_prompt + f"\nResponse: {rejected}"

        tokenized_chosen = tokenizer(chosen_response)
        tokenized_rejected = tokenizer(rejected_response)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["id"].append(id)

    return new_examples


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            prompt_messages = example["chosen"][:-1]
            # Prepend a system message if the first message is not a system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example
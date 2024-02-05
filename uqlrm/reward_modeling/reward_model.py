from peft import LoraConfig
from utils import build_model_quantization_config
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def build_reward_model(script_args):
    device_map, bnb_config, torch_dtype = build_model_quantization_config(script_args)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype=torch_dtype
    )

    print(model)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, trust_remote_code=True, truncation=True, max_length=script_args.seq_length)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    model.config.pad_token_id = model.config.eos_token_id # fix

    # PEFT config
    if script_args.use_peft and not script_args.inference:
        peft_config = LoraConfig(
                r=script_args.peft_lora_r,
                lora_alpha=script_args.peft_lora_alpha,
                target_modules=script_args.peft_lora_target_modules,
                bias="none",
                task_type="SEQ_CLS",
                modules_to_save=["scores"],
            )
    else:
        peft_config = None
    
    return model, tokenizer, peft_config
from peft import LoraConfig
from utils import build_model_quantization_config
from modules import RewardMLP, VariationalEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardModelFactory:
    def __init__(self):
        self.models = {
            'finetune_ens': self._create_finetune_ensembles,
            'adapters_ens': self._create_adapters_ensembles,
            'vatiational': self._create_variational_reward_model
        }

    def create(self, model_type):
        if model_type not in self.models:
            raise ValueError(f'Invalid model_type: {model_type}')
        return self.models[model_type]

    def _create_finetune_ensembles(self, script_args):
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

    def _create_adapters_ensembles(self, script_args):
        model = RewardMLP(input_size=4096, 
                      layers =script_args.layers,
                      activation_fn=script_args.activation_fn,
                      init_func=script_args.init_func,
                      weight_init=script_args.weight_init)
        return model, None, None

    def _create_variational_reward_model(self, script_args):
        model = VariationalEncoder(input_size=4096, 
                      layers=script_args.layers,
                      activation_fn=script_args.activation_fn,
                      init_func=script_args.init_func,
                      weight_init=script_args.weight_init)
        return model, None, None
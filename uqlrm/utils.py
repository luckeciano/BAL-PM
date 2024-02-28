import numpy as np
from typing import Dict
from transformers import TrainerCallback, BitsAndBytesConfig
import torch
from accelerate import Accelerator
from huggingface_hub import HfApi
import time


def softmax(x, temperature=0.5):
    e_x = np.exp(x / temperature)
    return e_x / e_x.sum(axis=0)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    print(model)

    
def compute_accuracy_with_inputs(eval_pred) -> Dict[str, float]:
    predictions, labels, inputs = eval_pred
    # Here, predictions is rewards_chosen and rewards_rejected.
    # We want to see how much of the time rewards_chosen > rewards_rejected.
    predictions = np.argmax(predictions, axis=1)

    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy, "ids": inputs}

def build_model_quantization_config(script_args):
    # Load the model
    device_map = {"": Accelerator().local_process_index}
    torch_dtype = torch.bfloat16

    if script_args.quantization_scheme == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif script_args.quantization_scheme == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        print("Loading model with no quantization!")
        device_map = None
        bnb_config = None
        torch_dtype = None
    
    return device_map, bnb_config, torch_dtype

# Callback to evaluate models with random weights
class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

class EvaluateAfterEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_evaluate = True
        control.should_save = True

class StopCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True

def push_predictions_to_hub(output_filedir, predictions_dataset_hub):
    api = HfApi()
    succeeded = False
    for i in range(3): # 3 retries
        if succeeded:
            break
        try:
            api.upload_folder(
                folder_path=output_filedir,
                path_in_repo=output_filedir,
                repo_id=predictions_dataset_hub,
                repo_type="dataset",
            )
            succeeded = True
        except Exception as e:
            succeeded = False
            print(f"Attempt {i+1} failed with error: {str(e)}")
            time.sleep(20) # Wait 20 seconds until next retry
            if i == 2:
                print("Operation failed after maximum number of retries.")
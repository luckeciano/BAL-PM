import numpy as np
from typing import Dict
from transformers import TrainerCallback, BitsAndBytesConfig
import torch
from accelerate import Accelerator
from huggingface_hub import HfApi
import time
from sklearn.neighbors import NearestNeighbors


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


def find_kth_nearest_dists(points, k, device, batch_size=1024):
    with torch.no_grad():
        # Initialize an empty tensor to store the indices
        indices = torch.empty(points.shape[0], dtype=torch.long).to(device)
        final_dists = torch.empty(points.shape[0], dtype=torch.double).to(device)

        # Compute pairwise distances and topk in batches
        for i in range(0, points.shape[0], batch_size):
            dists = torch.cdist(points[i:i+batch_size], points, compute_mode='donot_use_mm_for_euclid_dist')
            topk_dists, batch_indices = torch.topk(dists, k, largest=False)
            indices[i:i+batch_size] = batch_indices[:, k-1]
            final_dists[i:i+batch_size] = topk_dists[:, k-1]

    # Return the final dist
    return final_dists

def compute_batch_dists(pool_pts, batch_pts, device, batch_size=1024):
    with torch.no_grad():
        final_dists = torch.empty((pool_pts.shape[0], batch_pts.shape[0]), dtype=torch.float64).to(device)
        for i in range(0, pool_pts.shape[0], batch_size):
            dists = []
            for j in range(0, batch_pts.shape[0], batch_size):
                dists.append(torch.cdist(pool_pts[i:i+batch_size], batch_pts[j:j+batch_size], compute_mode='donot_use_mm_for_euclid_dist'))
            dists = torch.cat(dists, dim=1)  # concatenate along the batch dimension
            final_dists[i:i+batch_size] = dists
    
    return final_dists

def compute_nbatch(dists, points, batch_pt, device, batch_size=1024):
    with torch.no_grad():
        ds = dists.view(-1, 1)
        nbatches = torch.empty(points.shape[0], dtype=torch.long).to(device)
        # TODO review this code
        for i in range(0, points.shape[0], batch_size):
            batch_dist = torch.cdist(points[i:i+batch_size], batch_pt, compute_mode='donot_use_mm_for_euclid_dist') # 1024, 320
            n_batch = (batch_dist <= ds[i:i+batch_size]).double()  # 1024, 320
            n_batch = torch.sum(n_batch, dim=1)
            nbatches[i:i+batch_size] = n_batch
    
    return nbatches

def compute_batch_knns(k, pool_pts, batch_pts, device, batch_size=1024):
    with torch.no_grad():
        final_dists = torch.empty(pool_pts.shape[0], dtype=torch.double).to(device)
        for i in range(0, pool_pts.shape[0], batch_size):
            dists = []
            for j in range(0, batch_pts.shape[0], batch_size):
                dists.append(torch.cdist(pool_pts[i:i+batch_size], batch_pts[j:j+batch_size], compute_mode='donot_use_mm_for_euclid_dist'))
            dists = torch.cat(dists, dim=1)  # concatenate along the batch dimension
            topk_dists, _ = torch.topk(dists, k, largest=False, dim=1)  # compute topk along the batch dimension
            final_dists[i:i+batch_size] = topk_dists[:, k-1]
    
    return final_dists
    

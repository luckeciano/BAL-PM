from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import torch

@dataclass
class FrozenBackboneCollator:

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        ids = []
        
        for feature in features:
            ft_chosen = []
            ft_rejected = []
            total_fts = len(feature.keys())
            num_fts = total_fts // 2
            for i in range(num_fts):
                ft_chosen.append(feature[f'{i}'])
                ft_rejected.append(feature[f'{i + num_fts}'])
            features_chosen.append(
                ft_chosen
            )
            features_rejected.append(
                ft_rejected
            )
            ids.append(feature['id'])

        batch = {
            "input_chosen": torch.tensor(features_chosen, dtype=torch.float),
            "input_rejected": torch.tensor(features_rejected, dtype=torch.float),
            "id": torch.tensor(ids, dtype=torch.long),
            "return_loss": True,
        }
        return batch
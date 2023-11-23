from dataclasses import dataclass
from typing import Optional

from trl.trainer.training_configs import RewardConfig


@dataclass
class RewardConfigWithSavedPredictions(RewardConfig):
    """
    RewardConfig where you can optionally save model predictions in the HuggingFace hub.
    """

    push_predictions_to_hub: Optional[bool] = False

    predictions_dataset_hub: Optional[str] = None

    save_predictions_steps: Optional[int] = 25



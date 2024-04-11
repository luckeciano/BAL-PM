from dataclasses import dataclass
from typing import Optional

from trl.trainer import RewardConfig


@dataclass
class RewardConfigWithSavedPredictions(RewardConfig):
    """
    RewardConfig where you can optionally save model predictions in the HuggingFace hub.
    """

    push_predictions_to_hub: Optional[bool] = False

    predictions_dataset_hub: Optional[str] = None

    save_predictions_steps: Optional[int] = 25

    predictions_dir: Optional[str] = None

    regularized_loss: Optional[bool] = False

    lambda_regularization: Optional[float] = 0.01

    mc_dropout_realizations: Optional[int] = 5



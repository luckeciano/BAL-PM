from dataclasses import dataclass
from typing import Optional


@dataclass
class ActiveLearningConfig():
    """
    Active Learning Config.
    """

    initial_sample_size: Optional[int] = 64

    ensemble_size: Optional[int] = 8

    epoch_steps: Optional[int] = 1

    active_batch_size: Optional[int] = 64

    run_name: Optional[str] = "ActiveLearningRun"



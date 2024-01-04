from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ActiveLearningConfig():
    """
    Active Learning Config.
    """

    initial_sample_size: Optional[int] = 64

    ensemble_size: Optional[int] = 8

    epoch_steps: Optional[int] = 1

    steps_per_epoch: Optional[int] = 1

    active_batch_size: Optional[int] = 64

    pool_size: Optional[int] = 4096

    run_name: Optional[str] = "ActiveLearningRun"

    output_dir: Optional[str] = "./output"

    heuristic: Optional[str] = "epistemic"

    selection_strategy: Optional[str] = "rank"

    def to_dict(self):
        return asdict(self)



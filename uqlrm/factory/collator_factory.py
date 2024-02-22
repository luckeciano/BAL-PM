from collators import FrozenBackboneNumpyCollator, RewardDataCollatorWithPaddingAndIndices

class DataCollatorFactory:
    def __init__(self):
        self.models = {
            'reward_collator': self._create_reward_data_collator,
            'frozen_backbone_collator': self._create_frozen_backbone_collator,
        }

    def create(self, collator_type):
        if collator_type not in self.models:
            raise ValueError(f'Invalid model_type: {collator_type}')
        return self.models[collator_type]

    def _create_reward_data_collator(self, tokenizer, max_length):
        return RewardDataCollatorWithPaddingAndIndices(tokenizer, max_length=max_length)
    
    def _create_frozen_backbone_collator(self, tokenizer, max_length):
        return FrozenBackboneNumpyCollator()
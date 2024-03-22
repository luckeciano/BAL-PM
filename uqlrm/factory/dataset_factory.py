from dataset_utils import NumPyDataset, DataFrameDataset

class DatasetFactory:
    def __init__(self):
        self.models = {
            'numpy': self._create_numpy_dataset,
            'pandas': self._create_pandas_dataset,
        }

    def create(self, collator_type):
        if collator_type not in self.models:
            raise ValueError(f'Invalid model_type: {collator_type}')
        return self.models[collator_type]

    def _create_numpy_dataset(self, data):
        return NumPyDataset(data)
    
    def _create_pandas_dataset(self, data):
        return DataFrameDataset(data)
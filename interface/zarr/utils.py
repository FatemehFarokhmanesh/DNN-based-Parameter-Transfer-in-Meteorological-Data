from collections import namedtuple

from torch.utils.data import Dataset

NormalizedVariableData = namedtuple('NormalizedData', ['data', 'scales'])


class MergedData(Dataset):

    def __init__(self, *datasets: Dataset):
        assert len(datasets) > 0
        self.datasets = datasets
        num_samples = len(datasets[0])
        for d in datasets:
            assert len(d) == num_samples

    def __getitem__(self, item):
        return tuple(d[item] for d in self.datasets)

    def __len__(self):
        return len(self.datasets[0])
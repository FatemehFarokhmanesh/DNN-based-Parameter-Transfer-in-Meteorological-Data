import os
from scratch.example_filehandler import NumpyDataStorage
from torch.utils.data import DataLoader
from datetime import datetime
import torch

dataset = NumpyDataStorage(os.path.join('data', 'pytorch', 't_850', 'samples', '1980'))
batch_size = 20
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

start_time = datetime.now()

for i, batch in enumerate(dataloader):
    print('{}: {}'.format(i, torch.mean(batch).item()))
    if i == 150:
        break

end_time = datetime.now()

print('Total time spent for processing {} data: {}'.format(
        dataset.__class__.__name__, (end_time - start_time).total_seconds())
    )
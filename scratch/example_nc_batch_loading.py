import os
import numpy as np
from scratch.example_filehandler import NetCDFDataStorage
from datetime import datetime
import torch

dataset = NetCDFDataStorage(os.path.join('data', 'netcdf', 'T850'))
batch_size = 20

idx = np.arange(len(dataset))
# np.random.shuffle(idx)
batches = np.array_split(idx, np.ceil(len(idx) / batch_size))

start_time = datetime.now()

for i, batch in enumerate(batches):
    data = dataset.get_data(batch)
    print('{}: {}'.format(i, torch.mean(data).item()))
    if i == 150:
        break

end_time = datetime.now()

print('Total time spent for processing {} data: {}'.format(
        dataset.__class__.__name__, (end_time - start_time).total_seconds())
    )
import os
import numpy as np
import xarray as xr
import torch
from datetime import datetime, timedelta


num_samples = 365 * 24
domain_size = (128, 256)

base_directory = '.'
folders = {'np': 'numpy', 'xr': 'xarray', 'pt': 'numpy'}
for f in folders:
    folders.update({f: os.path.join(base_directory, folders[f])})
    if not os.path.isdir(folders[f]):
        os.makedirs(folders[f])

for y in range(2001, 2002):
    data = np.random.randn(num_samples, *domain_size)
    # Store numpy and torch files
    dates = []
    samples = []
    d = datetime(y, 1, 1, 0)
    new_datetime_string = d.strftime('%Y-%m-%d-%H')
    i = 0
    while i < num_samples:
        datetime_string = new_datetime_string
        print(datetime_string)
        np.save(os.path.join(folders['np'], datetime_string + '.npy'), data[i])
        torch.save(torch.tensor(data[i]), os.path.join(folders['pt'], datetime_string + '.pt'))
        dates.append(d)
        samples.append(i)
        d = d + timedelta(hours=1)
        new_datetime_string = d.strftime('%Y-%m-%d-%H')
        if datetime_string.split('-')[1] != new_datetime_string.split('-')[1]:
            print('Saving xarray data')
            xrds = xr.DataArray(
                data[np.array(samples)],
                coords={'time': dates, 'x': np.arange(domain_size[0]), 'y': np.arange(domain_size[1])},
                dims=['time', 'x', 'y']
            )
            xrds.to_netcdf(os.path.join(folders['xr'], '{}.nc'.format('-'.join(datetime_string.split('-')[:2]))))
            dates = []
            samples = []
        i += 1

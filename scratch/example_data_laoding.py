import os
from scratch.example_filehandler import PyTorchDataStorage, NumpyDataStorage, NetCDFDataStorage
from datetime import datetime


def process_dataset(dataset):
    print('Processing dataset {}'.format(dataset.__class__.__name__))
    start_time = datetime.now()

    for i in range(len(dataset)):
        data = dataset[i]
        processed = data ** 2

    end_time = datetime.now()

    print('Total time spent for processing {} data: {}'.format(
        dataset.__class__.__name__, (end_time - start_time).total_seconds())
    )


def process_data_batches(dataset):
    print('Processing dataset {}'.format(dataset.__class__.__name__))
    start_time = datetime.now()

    for i in range(len(dataset)):
        data = dataset[i]
        processed = data ** 2

    end_time = datetime.now()

    print('Total time spent for processing {} data: {}'.format(
        dataset.__class__.__name__, (end_time - start_time).total_seconds())
    )


nc_dataset = NetCDFDataStorage(os.path.join('data', 'netcdf', 'T850'))
pt_dataset = PyTorchDataStorage(os.path.join('data', 'numpy', 't_850_pt', 'samples', '1979'))
np_dataset = NumpyDataStorage(os.path.join('data', 'numpy', 't_850_np', 'samples', '1979'))

# assert len(nc_dataset) == len(pt_dataset), '[ERROR] {} != {}'.format(len(nc_dataset), len(pt_dataset))

process_dataset(pt_dataset)
process_dataset(np_dataset)
process_dataset(nc_dataset)

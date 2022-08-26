import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from datetime import datetime


class PyTorchDataStorage(Dataset):

    DATETIME_FORMAT = '%Y-%m-%d-%H'

    def __init__(self, path):
        super(PyTorchDataStorage, self).__init__()
        self.path = os.path.abspath(path)
        self.contents = [f for f in sorted(os.listdir(path)) if self._is_valid_file_name(f)]
        self._integer_index = {i: os.path.join(self.path, f) for i, f in enumerate(self.contents)}
        self._date_index = {self._file_name_to_datetime(f): os.path.join(self.path, f) for f in self.contents}

    def _file_name_to_datetime(self, f):
        return np.datetime64(datetime.strptime(f.split('.')[0], self.DATETIME_FORMAT))

    def _is_valid_file_name(self, f):
        if not f.endswith('.pt'):
            return False
        f_split = f.split('.')
        if len(f_split) > 2:
            return False
        try:
            date = self._file_name_to_datetime(f)
        except:
            return False
        return True

    def __getitem__(self, item):
        input_type = type(item)
        if input_type == np.datetime64:
            return torch.load(self._date_index[item])
        elif input_type == int:
            return torch.load(self._integer_index[item])
        elif input_type == datetime:
            return torch.load(self._date_index[np.datetime64(item)])
        else:
            raise ValueError('[ERROR] Item identifiers must be of type integer, numpy.datetime64 or datetime.datetime.')

    def __len__(self):
        return len(self.contents)


class NumpyDataStorage(Dataset):

    DATETIME_FORMAT = '%Y-%m-%d-%H'

    def __init__(self, path):
        super(NumpyDataStorage, self).__init__()
        self.path = os.path.abspath(path)
        self.contents = [f for f in sorted(os.listdir(path)) if self._is_valid_file_name(f)]
        self._integer_index = {i: os.path.join(self.path, f) for i, f in enumerate(self.contents)}
        self._date_index = {self._file_name_to_datetime(f): os.path.join(self.path, f) for f in self.contents}

    def _file_name_to_datetime(self, f):
        return np.datetime64(datetime.strptime(f.split('.')[0], self.DATETIME_FORMAT))

    def _is_valid_file_name(self, f):
        if not f.endswith('.npy'):
            return False
        f_split = f.split('.')
        if len(f_split) > 2:
            return False
        try:
            date = self._file_name_to_datetime(f)
        except:
            return False
        return True

    def __getitem__(self, item):
        input_type = type(item)
        if input_type == np.datetime64:
            data = np.load(self._date_index[item])
        elif input_type == int:
            data = np.load(self._integer_index[item])
        elif input_type == datetime:
            data = np.load(self._date_index[np.datetime64(item)])
        else:
            raise ValueError('[ERROR] Item identifiers must be of type integer, numpy.datetime64 or datetime.datetime.')
        return torch.tensor(data)

    def __len__(self):
        return len(self.contents)


class NetCDFDataStorage(Dataset):
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.data = xr.open_mfdataset(self._list_path_directory())
        self.data =  self.data.data_vars[list(self.data.data_vars.keys())[0]]
        self._time_stamps = self.data.time.values

    def _list_path_directory(self):
        return [os.path.join(self.path, f) for f in sorted(os.listdir(self.path)) if f.endswith('.nc')]

    def __getitem__(self, item):
        input_type = type(item)
        if input_type == np.datetime64:
            data = self.data.sel(time=[item])
        elif input_type == int:
            data = self.data.sel(time=[self._time_stamps[item]])
        elif input_type == datetime:
            data = self.data.sel(time=[np.datetime64(item)])
        else:
            raise ValueError('[ERROR] Item identifiers must be of type integer, numpy.datetime64 or datetime.datetime.')
        return torch.tensor(data.values)

    def get_data(self, idx):
        time_stamps = self._time_stamps[idx]
        data = self.data.sel(time=time_stamps).values
        return torch.tensor(data)

    def __len__(self):
        return len(self._time_stamps)


class NetCDFDaskStorage(Dataset):
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.data = xr.open_mfdataset(self._list_path_directory())
        self.data = self.data.data_vars[list(self.data.data_vars.keys())[0]]
        self._dask_array = self.data.data
        self._time_stamps = self.data.time.values
        self._sample_index = {ts: i for i, ts in enumerate(self._time_stamps)}

    def _list_path_directory(self):
        return [os.path.join(self.path, f) for f in sorted(os.listdir(self.path)) if f.endswith('.nc')]

    def __getitem__(self, item):
        input_type = type(item)
        if input_type == np.datetime64:
            data = self._dask_array[self._sample_index[item]]
        elif input_type == int:
            data = self._dask_array[item]
        elif input_type == datetime:
            data = self._dask_array[self._sample_index[np.datetime64(item)]]
        else:
            raise ValueError('[ERROR] Item identifiers must be of type integer, numpy.datetime64 or datetime.datetime.')
        return torch.tensor(data.compute())

    def get_data(self, idx):
        data = self._dask_array[idx]
        data = data.compute()
        return torch.tensor(data)

    def __len__(self):
        return len(self._time_stamps)
import os
import xarray as xr
import numpy as np
from collections import namedtuple
import torch
from torch.utils.data import Dataset

from .transforms import Transform
from utils import ProgressBar
import pandas as pd

TEMPORAL_RESOLUTION = np.timedelta64(1, 'h')


class TimeRange(object):
    def __init__(self, start_date, end_date, include_end_date=False, format=None):
        self.start_date = self._parse_date_input(start_date, 'start_date', format)
        self.end_date = self._parse_date_input(end_date, 'start_date', format)
        self.include_end_date = include_end_date

    def _parse_date_input(self, date_input, date_name, format):
        if isinstance(date_input, np.datetime64):
            return date_input
        elif isinstance(date_input, str):
            return pd.to_datetime(date_input, format=format).to_numpy()
        else:
            raise Exception('[ERROR] Encountered invalid input for date variable <start_date>.')

    def get_timesteps(self, resolution):
        assert isinstance(resolution, np.timedelta64)
        end_date = self.end_date
        if self.include_end_date:
            end_date = end_date + resolution
        return np.arange(self.start_date, end_date, resolution)


class ParameterDataIdentifier(object):
    def __init__(self, source_parameter):
        self.source_parameter = source_parameter

    def extends(self, other):
        raise NotImplementedError()

    def source_consistent_with(self, other):
        return self.source_parameter.full_name() == other.source_parameter.full_name()


class ConstantDataIdentifier(ParameterDataIdentifier):
    def __init__(self, source_parameter):
        assert isinstance(source_parameter, ConstantData)
        super(ConstantDataIdentifier, self).__init__(source_parameter)

    def extends(self, other):
        return False


class VariableDataIdentifier(ParameterDataIdentifier):
    def __init__(self, source_parameter, samples):
        assert isinstance(source_parameter, VariableData)
        super(VariableDataIdentifier, self).__init__(source_parameter)
        assert isinstance(samples, np.ndarray) and len(samples.shape) == 1
        self.samples = samples

    def extends(self, other):
        if not isinstance(other, VariableDataIdentifier):
            return False
        if not self.source_consistent_with(other):
            return False
        if not self._extends_samples_of(other):
            return False
        return True

    def _extends_samples_of(self, other):
        other_samples = other.samples
        min_date = np.min(other_samples)
        data_range = np.arange(min_date, np.max(other_samples) + TEMPORAL_RESOLUTION, TEMPORAL_RESOLUTION)
        self_samples_idx = ((self.samples - min_date) / TEMPORAL_RESOLUTION).astype(int)
        if np.any(self_samples_idx < 0) or np.any(self_samples_idx > len(data_range)):
            return False
        other_samples_idx = ((other_samples - min_date) / TEMPORAL_RESOLUTION).astype(int)
        data_mask = np.zeros_like(data_range).astype(bool)
        data_mask[other_samples_idx] = True
        data_mask[self_samples_idx] = False
        return not np.any(data_mask)

    def get_missing_samples(self, other):
        other_samples = other.samples
        self_samples = self.samples
        min_date = np.min(self_samples)
        data_range = np.arange(min_date, np.max(self_samples) + TEMPORAL_RESOLUTION, TEMPORAL_RESOLUTION)
        data_mask = np.zeros_like(data_range).astype(bool)
        self_samples_idx = ((self_samples - min_date) / TEMPORAL_RESOLUTION).astype(int)
        other_samples_idx = ((other_samples - min_date) / TEMPORAL_RESOLUTION).astype(int)
        if np.any(other_samples_idx < 0) or np.any(other_samples_idx > len(data_range)):
            raise Exception("[ERROR] Requested samples inconsistent with previously seen training samples.")
        data_mask[self_samples_idx] = True
        data_mask[other_samples_idx] = False
        return data_range[data_mask]

    def include(self, other):
        self.samples = np.concatenate([self.samples, other.samples], axis=0)
        return self


dataset_reference = namedtuple('dataset_reference', ['dataset', 'group_key'])


class ParameterData(Dataset):
    def __init__(self, name, source, selectors=None, transforms=None):
        assert isinstance(name, str)
        self.name = name
        assert isinstance(source, DataStorage)
        self.source = source
        self.selectors = None
        if selectors is not None:
            self._parse_selectors(selectors)
            self.source.restrict_to(**selectors)
        self.transforms = []
        if transforms is not None:
            self.add_transforms(transforms)
        self._dataset_reference = None

    def _parse_selectors(self, selectors):
        assert isinstance(selectors, dict)
        for key in selectors:
            assert self.source.is_selectable_by(key), \
                '[ERROR] Source <{}> does not support selector {}.'.format(self.source, key)
        self.selectors = selectors

    def add_transforms(self, transforms):
        transforms = self._parse_transforms(transforms)
        if transforms:
            self.transforms += transforms
        return self

    @staticmethod
    def _parse_transforms(transforms):
        if transforms is None:
            return []
        if isinstance(transforms, Transform):
            transforms = [transforms]
        else:
            try:
                transforms = list(transforms)
            except:
                raise Exception('[ERROR] Transforms must be given as single Transform objects, lists or tuples thereof.')
            for item in transforms:
                assert isinstance(item, Transform)
        return transforms

    def reset_transforms(self, transforms=None):
        self.transforms = []
        if transforms is not None:
            self.add_transforms(transforms)
        return self

    def fit_transforms(self):
        if not self.transforms:
            print('[INFO] No transforms to fit for variable <{}>.'.format(self.full_name()))
        else:
            print('[INFO] Fitting transforms for variable <{}>.'.format(self.full_name()))
            self._fit_transforms()

    def _fit_transforms(self):
        raise NotImplementedError()

    def full_name(self):
        if self._dataset_reference is None:
            return self.name
        else:
            return '.'.join([self._dataset_reference.group_key, self.name])

    def is_time_variate(self):
        return self.source.is_time_variate()

    def load(self):
        self.source.load()
        return self

    def restrict_to(self, **kwargs):
        raise NotImplementedError()

    def get_valid_coordinates(self, dim=None):
        if dim is None:
            return {dim_name: self.get_valid_coordinates(dim=dim_name) for dim_name in self.source.dimension_names()}
        return getattr(self.source.data, dim).values if dim in self.source.data.dims else None

    def set_dataset_reference(self, dataset, group_key):
        self._dataset_reference = dataset_reference(dataset, group_key)

    def provides_time_stamps(self, time_stamps):
        return self.source.provides_time_stamps(time_stamps)

    def channel_count(self):
        raise NotImplementedError()

    def get_data_batch(self, time_stamps):
        raise NotImplementedError()

    def __getitem__(self, time_stamps):
        time_stamps = self._time_stamps_to_nparray(time_stamps)
        return self.get_data_batch(time_stamps)

    @staticmethod
    def _time_stamps_to_nparray(time_stamps):
        input_type = type(time_stamps)
        if input_type != np.ndarray:
            if input_type == np.datetime64:
                time_stamps = np.array([time_stamps])
            else:
                try:
                    time_stamps = np.array(time_stamps).astype(np.datetime64)
                except:
                    raise Exception(
                        "[ERROR] Input of type <{}> is not a valid time stamp specification.".format(input_type)
                    )
        return time_stamps

    def __len__(self):
        raise NotImplementedError()

    @staticmethod
    def _item_to_ndarray(item):
        input_type = type(item)
        if input_type != np.ndarray:
            if input_type == np.datetime64:
                item = np.array([item])
            else:
                try:
                    item = np.array(item)
                except:
                    raise Exception("[ERROR] {} is not a valid item specification.".format(item))
        return item


class ConstantData(ParameterData):
    def __init__(self, name, source, selectors=None, transforms=None):
        super(ConstantData, self).__init__(name, source, selectors=selectors, transforms=transforms)
        assert not source.is_time_variate()

    def _fit_transforms(self):
        for i, current_transform in enumerate(self.transforms):
            if current_transform.is_data_adaptive():
                training_history = current_transform.get_training_history()
                if training_history is None:
                    data = self.source.get_data()
                    previous_transforms = self.transforms[:i]
                    for pt in previous_transforms:
                        data = pt(data)
                    current_transform.update_fit(self.source.get_data(), ConstantDataIdentifier(self))
                else:
                    data_identifier = ConstantDataIdentifier(self)
                    assert (
                            isinstance(training_history, ConstantDataIdentifier)
                            and
                            (training_history.source_consistent_with(data_identifier))
                    ), "[ERROR] Transforms cannot be fitted to more than one constant data source."
                    print("[INFO] Transform already fitted.")
        return self

    def restrict_to(self, **kwargs):
        self.source.restrict_to(**kwargs)
        return self

    def channel_count(self):
        return self.source.channel_count()

    def get_data_batch(self, time_stamps):
        data = self.source.get_data()
        if self.transforms:
            for transform in self.transforms:
                data = transform(data)
        try:
            num_samples = len(time_stamps)
        except TypeError:
            num_samples = 1
        if num_samples > 1:
            data = data.expand(num_samples, -1, -1, -1)
        return data

    def __getitem__(self, item):
        return self.get_data_batch(self._item_to_ndarray(item))

    def __len__(self):
        return 1


class VariableData(ParameterData):
    def __init__(self, name, source, delay_steps=None, delay=None, selectors=None, transforms=None):
        super(VariableData, self).__init__(name, source, selectors=selectors, transforms=transforms)
        assert isinstance(source, DataStorage) and source.is_time_variate()
        if delay_steps is not None:
            assert isinstance(delay_steps, int)
            assert delay_steps >= 0
            assert isinstance(delay, int)
            assert delay > 0
            delay = self._parse_time_delta(delay)
        else:
            delay_steps = 0
            delay = None
        self.delay_steps = delay_steps
        self.delay = delay

    @staticmethod
    def _parse_time_delta(delta):
        assert type(delta) == int, '[ERROR] Time differences must be given as integer.'
        return delta * TEMPORAL_RESOLUTION

    def _fit_transforms(self):
        if self._dataset_reference is not None:
            ref = self._dataset_reference
            sample_time_stamps = ref.dataset.get_transform_time_stamps(ref.group_key)
        else:
            sample_time_stamps = self.source.get_valid_time_stamps()
        for i, current_transform in enumerate(self.transforms):
            if current_transform.requires_fit:
                training_history = current_transform.get_training_history()
                previous_transforms = self.transforms[:i]
                if training_history is None:
                    self._fit_transform_to_samples(current_transform, sample_time_stamps, previous_transforms)
                else:
                    data_identifier = VariableDataIdentifier(self, sample_time_stamps)
                    if data_identifier.extends(training_history):
                        missing_samples = training_history.filter_missing_samples(sample_time_stamps)
                        self._fit_transform_to_samples(current_transform, missing_samples, previous_transforms)
                    else:
                        raise Exception("[ERROR] Encountered invalid sample set for fitting transforms")
        return self

    def _fit_transform_to_samples(self, transform, samples, previous_transforms):
        chunks = self.source.apply_chunking(samples)
        pbar = ProgressBar(len(chunks))
        for sample_chunk in chunks:
            data = self.source.get_data(time=sample_chunk)
            for pt in previous_transforms:
                data = pt(data)
            transform.update_fit(data, VariableDataIdentifier(self, sample_chunk))
            pbar.step()

    def _include_delayed_samples(self, time_stamps):
        if self.delay_steps == 0:
            return time_stamps
        else:
            out = np.array([(time_stamps - lag * self.delay) for lag in range(self.delay_steps + 1)])
            out = np.ravel(out)
            return out

    def restrict_to(self, **kwargs):
        if self.delay_steps > 0 and 'time' in kwargs:
            time_stamps = kwargs['time']
            time_stamps = self._include_delayed_samples(time_stamps)
            kwargs.update({'time': np.unique(time_stamps)})
        self.source.restrict_to(**kwargs)
        return self

    def get_valid_coordinates(self, dim=None):
        if dim is None:
            return {dim_name: self.get_valid_coordinates(dim=dim_name) for dim_name in self.source.dimension_names()}
        elif dim == 'time':
            return self.get_valid_time_stamps()
        else:
            return getattr(self.source.data, dim).values if dim in self.source.dimension_names() else None

    def get_valid_time_stamps(self):
        source_time_stamps = self.source.get_valid_time_stamps()
        if self.delay_steps > 0:
            min_time, max_time = np.min(source_time_stamps), np.max(source_time_stamps)
            extended_time_stamps = TimeRange(min_time, max_time, include_end_date=True).get_timesteps(TEMPORAL_RESOLUTION)
            delay_freq = int(self.delay / TEMPORAL_RESOLUTION)
            mask = np.zeros(len(extended_time_stamps))
            mask[((source_time_stamps - min_time) / TEMPORAL_RESOLUTION).astype(int)] = 1
            pattern = np.zeros(self.delay_steps * delay_freq + 1)
            pattern[::delay_freq] = 1
            valid = (np.convolve(mask, pattern, mode='full') == (self.delay_steps + 1))[:len(extended_time_stamps)]
            source_time_stamps = extended_time_stamps[valid]
        return source_time_stamps

    def provides_time_stamps(self, time_stamps):
        if self.delay_steps > 0:
            time_stamps = np.unique(self._include_delayed_samples(time_stamps))
        return self.source.provides_time_stamps(time_stamps)

    def channel_count(self):
        return self.source.channel_count() * (self.delay_steps + 1)

    def get_data_batch(self, time_stamps):
        time_stamps = self._include_delayed_samples(time_stamps)
        data = self.source.get_data(time=time_stamps)
        if self.transforms:
            for transform in self.transforms:
                data = transform(data)
        if self.delay_steps > 0:
            chunks = torch.chunk(data, self.delay_steps + 1, dim=0)
            data = torch.cat(chunks, dim=1)
        return data

    def __getitem__(self, item):
        return self.get_data_batch(self._item_to_ndarray(item))

    def __len__(self):
        return len(self.get_valid_time_stamps())


class DataStorage(Dataset):
    def __init__(self, data):
        self.data = data

    def dimension_names(self):
        return tuple(self.data.dims)

    def is_time_variate(self):
        return 'time' in self.dimension_names()

    def provides_time_stamps(self, time_stamps):
        if not self.is_time_variate():
            return True
        available_time_stamps = self.get_valid_time_stamps()
        required_min_date = np.min(time_stamps)
        required_max_date = np.max(time_stamps)
        required_time_range = TimeRange(required_min_date, required_max_date, include_end_date=True).get_timesteps(TEMPORAL_RESOLUTION)
        mask_required_dates = np.zeros_like(required_time_range).astype(bool)
        idx_required_dates = ((time_stamps - required_min_date) / TEMPORAL_RESOLUTION).astype(int)
        mask_required_dates[idx_required_dates] = True
        mask_available_dates = np.zeros_like(required_time_range).astype(bool)
        idx_available_dates = ((available_time_stamps - required_min_date) / TEMPORAL_RESOLUTION).astype(int)
        idx_available_dates = idx_available_dates[np.logical_and(idx_available_dates >= 0, idx_available_dates < len(mask_available_dates))]
        mask_available_dates[idx_available_dates] = True
        return np.all(np.logical_and(mask_available_dates, mask_required_dates)[mask_required_dates])

    def get_valid_coordinates(self, dim=None):
        if dim is None:
            return {dim_name: self.get_valid_coordinates(dim=dim_name) for dim_name in self.dimension_names()}
        return getattr(self.data, dim).values if dim in self.dimension_names() else None

    def get_valid_time_stamps(self):
        return self.get_valid_coordinates('time')

    def get_data(self, **kwargs):
        if len(kwargs):
            data = self.data.sel(**kwargs)
        else:
            data = self.data
        data = torch.tensor(data.values)
        if not self.is_time_variate():
            data = data.unsqueeze(dim=0)
        dim = len(data.shape)
        if dim == 3:
            return data.unsqueeze(dim=1)
        elif dim == 4:
            return data
        elif dim > 4:
            return torch.flatten(data, start_dim=1, end_dim=-3)
        else:
            raise NotImplementedError()

    def restrict_to(self, **kwargs):
        self.data = self.data.sel(**kwargs)
        return self

    def load(self):
        self.data.load()
        return self

    def channel_count(self):
        start_dim = 1 if self.is_time_variate() else 0
        channel_dims = self.data.shape[start_dim:-2]
        if not len(channel_dims):
            channel_dims = (1,)
        return np.prod(channel_dims)

    def apply_chunking(self, time_stamps):
        raise NotImplementedError()

    def is_selectable_by(self, key):
        return self.get_valid_coordinates(key) is not None


class NetCDFStorage(DataStorage):
    def __init__(self, source_name, source_path, ext='.nc', chunk_size=None, parallel=True):
        self.source_name = source_name
        self.source_path = os.path.abspath(source_path)
        self.ext = ext
        self.chunk_size = chunk_size
        self.parallel = parallel
        super(NetCDFStorage, self).__init__(self._read_storage())

    def _read_storage(self):
        if self.source_path.endswith(self.ext):
            data = self._read_data_file(
                self.source_path, chunk_size=self.chunk_size
            )
        elif os.path.isdir(self.source_path):
            data = self._read_data_directory(
                self.source_path, ext=self.ext, chunk_size=self.chunk_size, parallel=self.parallel
            )
        else:
            raise Exception(
                '[ERROR] <{}> is not a valid directory name or data file with extension {}.'.format(
                    self.source_path, self.ext
                )
            )
        return getattr(data, self.source_name)

    @staticmethod
    def _read_data_file(path, chunk_size=None):
        data = xr.open_dataset(path, chunks=chunk_size)
        return data

    @staticmethod
    def _read_data_directory(directory, ext='.nc', chunk_size=None, parallel=True):
        data_files = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith(ext)]
        assert len(data_files), '[ERROR] Directory {} does not contain valid data files.'.format(directory)
        data = xr.open_mfdataset(data_files, chunks=chunk_size, parallel=parallel)
        return data

    def apply_chunking(self, time_stamps):
        if self.chunk_size is None or len(time_stamps) <= self.chunk_size['time']:
            return [time_stamps]
        else:
            return np.array_split(time_stamps, np.ceil(len(time_stamps) / self.chunk_size['time']))


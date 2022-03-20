import numpy as np
from collections import namedtuple, OrderedDict
import torch
from torch.utils.data import Dataset, IterableDataset

from .sources import ParameterData, ConstantData, VariableData, TEMPORAL_RESOLUTION, TimeRange


data_group_setting = namedtuple('data_group_setting', ['variables', 'constants', 'lead_time'])


class WeatherBenchData(Dataset):
    def __init__(self, data_groups, lead_times=None, time_range=None, fit_transforms=False):
        print('[INFO] Creating WeatherBench dataset.')
        super(WeatherBenchData, self).__init__()
        assert isinstance(data_groups, dict) and len(data_groups) > 0
        if lead_times is not None:
            assert isinstance(lead_times, dict)
        else:
            lead_times = {}
        self.data_groups = self._parse_data_items(data_groups, lead_times)
        if time_range is not None:
            assert isinstance(time_range, TimeRange), '[ERROR] If given, a time range must be specified as TimeRange object.'
            valid_time_stamps = time_range.get_timesteps(TEMPORAL_RESOLUTION)
            self._verify_data_availability(valid_time_stamps)
        else:
            valid_time_stamps = self._collect_valid_time_stamps()
        self.valid_time_stamps = valid_time_stamps
        self._restrict_data_to_valid_time_stamps()
        if fit_transforms:
            print('[INFO] Fitting transforms.')
            self.fit_transforms()

    def _parse_data_items(self, data_groups, lead_times):
        print('[INFO] Parsing data groups.')
        self._check_key_compatibility(data_groups, lead_times)
        parsed_data_groups = OrderedDict({})
        for group_key in data_groups:
            print('[INFO] Parsing group <{}>'.format(group_key))
            group_items = data_groups[group_key]
            group_variables = OrderedDict({})
            group_constants = OrderedDict({})
            if not isinstance(group_items, (list, tuple)):
                group_items = [group_items]
            for item in group_items:
                assert isinstance(item, ParameterData)
                item.set_dataset_reference(self, group_key)
                item_name = item.name
                if item_name in group_variables.keys():
                    raise Exception(self._unique_data_name_error(item.full_name()))
                if isinstance(item, VariableData):
                    print('[INFO] Listing variable data source <{}>.'.format(item.full_name()))
                    group_variables.update({item_name: item})
                elif isinstance(item, ConstantData):
                    print('[INFO] Listing constant data source <{}>.'.format(item.full_name()))
                    group_constants.update({item_name: item})
                else:
                    raise Exception()
            if group_key in lead_times:
                try:
                    group_lead_time = np.timedelta64(lead_times[group_key], 'h')
                except:
                    raise Exception(self._invalid_lead_time_error(lead_times, group_key))
            else:
                group_lead_time = self._lead_time_default()
            parsed_data_groups.update({group_key: data_group_setting(group_variables, group_constants, group_lead_time)})
        return parsed_data_groups

    def _check_key_compatibility(self, data_groups, lead_times):
        data_keys = list(data_groups.keys())
        lead_time_keys = list(lead_times.keys())
        unused_keys = list(set(lead_time_keys) - set(data_keys))
        if len(unused_keys) > 0:
            raise Exception(
                "[ERROR] The following keys of dictionary <lead_times> don't have an associated enry in dictionary <data_groups>: }".format(unused_keys)
            )

    def _verify_data_availability(self, time_stamps):
        print('[INFO] Verifying availability of required data.')
        for group_key in self.data_groups:
            data_group = self.data_groups[group_key]
            group_variables = data_group.variables.values()
            group_time_stamps = time_stamps + data_group.lead_time
            for item in group_variables:
                assert item.provides_time_stamps(group_time_stamps), self._missing_data_error(item.full_name())

    def _collect_valid_time_stamps(self):
        print('[INFO] collecting valid time stamps.')
        running_min_date = None
        running_max_date = None
        all_group_time_stamps = []
        for group_key in self.data_groups:
            data_group = self.data_groups[group_key]
            group_variables = data_group.variables.values()
            group_lead_time = data_group.lead_time
            for item in group_variables:
                group_time_stamps = item.get_valid_time_stamps() - group_lead_time
                if running_min_date is None:
                    running_min_date = np.min(group_time_stamps)
                else:
                    running_min_date = np.maximum(running_min_date, np.min(group_time_stamps))
                if running_max_date is None:
                    running_max_date = np.max(group_time_stamps)
                else:
                    running_max_date = np.minimum(running_max_date, np.max(group_time_stamps))
                all_group_time_stamps.append(group_time_stamps)
        time_range = TimeRange(running_min_date, running_max_date, include_end_date=True)
        all_time_stamps = time_range.get_timesteps(TEMPORAL_RESOLUTION)
        mask = np.zeros((len(all_time_stamps), len(all_group_time_stamps)))
        for i, group_time_stamps in enumerate(all_group_time_stamps):
            group_mask = np.logical_and(group_time_stamps >= running_min_date, group_time_stamps <= running_max_date)
            group_idx = ((group_time_stamps[group_mask] - running_min_date) / TEMPORAL_RESOLUTION).astype(int)
            mask[i, group_idx] = True
        valid_time_stamps = all_time_stamps[np.all(mask, axis=-1)]
        return valid_time_stamps

    def _restrict_data_to_valid_time_stamps(self):
        for group_key in self.data_groups:
            data_group = self.data_groups[group_key]
            group_variables = data_group.variables.values()
            group_time_stamps = self.valid_time_stamps + data_group.lead_time
            for item in group_variables:
                item.restrict_to(time=group_time_stamps)

    @staticmethod
    def _unique_data_name_error(name):
        return "[ERROR] Data item names must be unique, but name {} was used multiple times.".format(name)

    @staticmethod
    def _invalid_lead_time_error(lead_times, group_key):
        return "[ERROR] Lead-time specification <{}> for data group key <{}> is invalid.".format(lead_times[group_key], group_key)

    @staticmethod
    def _missing_data_error(name):
        return "[ERROR] Data item <{}> is unable to provide the full set of requested time stamps."

    @staticmethod
    def _lead_time_default():
        return np.timedelta64(0, 'h')

    def get_valid_time_stamps(self):
        return self.valid_time_stamps

    def __len__(self):
        return len(self.valid_time_stamps)

    def __getitem__(self, item):
        input_type = type(item)
        if input_type != np.ndarray:
            if input_type == int:
                item = np.ndarray([item])
            else:
                try:
                    item = np.array(item)
                except Exception:
                    raise Exception("[ERROR] {}  is not a valid item specification.".format(item))
        current_time_stamps = self.valid_time_stamps[item]
        all_data = []
        for data_group in self.data_groups.values():
            group_data = []
            group_time_stamps = current_time_stamps + data_group.lead_time
            group_data += [item.get_data_batch(group_time_stamps) for item in data_group.variables.values()]
            group_data += [item.get_data_batch(group_time_stamps) for item in data_group.constants.values()]
            if group_data:
                group_data = torch.cat(group_data, dim=1)
            else:
                group_data = self._empty_data_default()
            all_data.append(group_data)
        return tuple(all_data)

    @staticmethod
    def _empty_data_default():
        return []

    def load(self, mode='all', **kwargs):
        if mode.lower() in ['all', 'variables']:
            for data_group in self.data_groups.values():
                group_time_stamps = self.valid_time_stamps + data_group.lead_time
                for item in data_group.variables.values():
                    item.restrict_to(time=group_time_stamps, **kwargs)
                    item.load()
        if mode.lower() in ['all', 'constants']:
            for data_group in self.data_groups.values():
                for item in data_group.constants.values():
                    item.restrict_to(**kwargs)
                    item.load()
        return self

    def channel_count(self, mode=None):
        if mode is None:
            return {
                key: self.channel_count(mode=key)
                for key in self.data_groups
            }
        elif isinstance(mode, (list, tuple)):
            return {
                key: self.channel_count(mode=key)
                for key in mode
            }
        elif isinstance(mode, str):
            if mode in self.data_groups:
                all_items = []
                all_items += list(self.data_groups[mode].variables.values())
                all_items += list(self.data_groups[mode].constants.values())
                return np.sum([item.channel_count() for item in all_items])
            else:
                raise NotImplementedError('[ERROR] Channel mode <{}> unavailable.'.format(mode))
        else:
            raise NotImplementedError('[ERROR] Channel mode <{}> unavailable.'.format(mode))

    def get_transform_time_stamps(self, group_key):
        if group_key in self.data_groups:
            sample_time_stamps = self.valid_time_stamps + self.data_groups[group_key].lead_time
            return sample_time_stamps
        else:
            raise Exception('[ERROR] Dataset does not contain parameter group with key {}.'.format(group_key))
    
    def fit_transforms(self):
        for data_group in self.data_groups.values():
            for item in data_group.variables.values():
                item.fit_transforms()
            for item in data_group.constants.values():
                item.fit_transforms()
        return self
    
    def copy_transforms(self, other):
        for key in self.data_groups:
            self_group = self.data_groups[key]
            other_group = other.data_groups[key]
            for item in self_group.variables.values():
                other_item = other_group.variables[item.name]
                item.reset_transforms(other_item.transforms)
            for item in self_group.constants.values():
                other_item = other_group.constants[item.name]
                item.reset_transforms(other_item.transforms)
        return self

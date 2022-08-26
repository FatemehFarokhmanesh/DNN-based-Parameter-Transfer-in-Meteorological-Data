import os
import socket
from collections import namedtuple
from enum import Enum
import sys
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

from torch.utils.data import Dataset

from configs import ProjectConfigs
from utils import NormalizedVariableData



EXT_NETCDF = '.nc'
EXT_ZARR = '.zarr'

NETCDF_PATHS = {
    ProjectConfigs().SERVER_NAME: ProjectConfigs().DATA_PATH + 'netcdf'
}

ZARR_PATHS = {
    ProjectConfigs().SERVER_NAME: ProjectConfigs().DATA_PATH + 'zarr'
}


def _get_host_name():
    return socket.gethostname()


def get_netcdf_path():
    return NETCDF_PATHS[_get_host_name()]


def get_zarr_path():
    return ZARR_PATHS[_get_host_name()]


class Variable3d(Enum):
    TK = 'tk'
    U = 'u'
    V = 'v'
    W = 'w'
    Z = 'z'
    RH = 'rh'
    QV = 'qv'
    QH = 'qhydro'
    DBZ = 'dbz'


CoordinateSummary = namedtuple('CoordinateSummary', ['name', 'min', 'max'])


class CoordinateAxis(Enum):
    MEMBER = CoordinateSummary('member', 0, 1000)
    TIME = CoordinateSummary('time', 0, 6)
    LEVEL = CoordinateSummary('lev', 8, 20)
    LAT = CoordinateSummary('lat', 0, 352)
    LON = CoordinateSummary('lon', 0, 250)


class MultiVariableData2d(Dataset):

    DUMMY_DIM_NAME = 'dummy'
    MODE = 'zarr'

    def __init__(
            self, variables: List[Variable3d],
            selection=None, scales=None,
            base_path=None
    ):
        _all_axes = [v for v in Variable3d]
        self.variables = variables
        if selection is None:
            selection = {}
        self.selections = self._parse_selection(selection)
        if scales is not None:
            norm_dims = None
        else:
            norm_dims = [self.DUMMY_DIM_NAME] + \
                [ax.value.name for ax in [CoordinateAxis.LON, CoordinateAxis.LAT]]
        self.norm_dims = norm_dims
        if base_path is None:
            base_path = get_zarr_path() if self.MODE == 'zarr' else get_netcdf_path()
        self.base_path = base_path
        self.data = self._load_data(scales=scales)

    @staticmethod
    def _parse_selection(selection):

        def parsing(x, axis_name):
            ax = getattr(CoordinateAxis, axis_name).value
            max_idx = ax.max - ax.min
            if x is None:
                x = np.arange(max_idx)
            else:
                x = np.sort(np.array(x))
            assert np.all(x < max_idx), \
                f'[ERROR] some of the selections along dim {ax.value.name} are beyond the maximum allowed range (0 to {max_idx})'
            return x

        out = {}
        for ax in CoordinateAxis:
            sel = selection[ax] if ax in selection else None
            out[ax] = parsing(sel, ax.name)
        return out

    def _file_valid(self, file_name):
        if self.MODE == 'nc':
            ext = EXT_NETCDF
        elif self.MODE == 'zarr':
            ext = EXT_ZARR
        else:
            raise NotImplementedError()
        base_name, extension = os.path.splitext(file_name)
        member_idx = int(base_name[-4:]) - 1
        if extension == ext and member_idx in self.selections[CoordinateAxis.MEMBER]:
            return True
        return False

    def _load_data(self, scales=None):
        dir_list = os.listdir(self.base_path)
        files = sorted([f for f in dir_list if self._file_valid(f)])
        selections = {
            ax.value.name: ax.value.min + self.selections[ax]
            for ax in [CoordinateAxis.TIME, CoordinateAxis.LEVEL, CoordinateAxis.LAT, CoordinateAxis.LON]
        }
        drop_vars = [v.value for v in Variable3d if v not in self.variables]
        chunks = {v.value.name: 1 for v in [
            CoordinateAxis.TIME, CoordinateAxis.LEVEL]}
        data = [
            xr.open_dataset(
                os.path.join(self.base_path, f),
                drop_variables=drop_vars,
                chunks=chunks
            ).isel(selections) for f in files
        ]
        member_index = pd.Index(
            self.selections[CoordinateAxis.MEMBER], name=CoordinateAxis.MEMBER.value.name)
        data = xr.concat(data, member_index)
        stack_dims = {
            self.DUMMY_DIM_NAME: [
                ax.value.name
                for ax in [CoordinateAxis.MEMBER, CoordinateAxis.TIME, CoordinateAxis.LEVEL]
            ]
        }
        data = data.stack(**stack_dims)
        data = data.transpose(
            self.DUMMY_DIM_NAME, CoordinateAxis.LAT.value.name, CoordinateAxis.LON.value.name)
        if scales is None:

            def _compute_scales(v):
                vdata = data[v.value]
                mu = vdata.mean(dim=self.norm_dims).compute()
                std = vdata.std(dim=self.norm_dims, ddof=1).compute()
                return (mu, std)

            scales = {v: _compute_scales(v) for v in self.variables}

        def process_variable(v):
            vdata = data[v.value]
            mu, std = scales[v]
            return NormalizedVariableData(vdata, (mu, std))

        data = {v: process_variable(v) for v in self.variables}

        return data

    def output_channels(self):
        return len(self.variables)

    def __len__(self):
        return len(self.data[self.variables[0]].data[self.DUMMY_DIM_NAME])

    def get_scales(self):
        return {v: self.data[v].scales for v in self.variables}

    def __getitem__(self, item):
        data = [self._get_variable_item(v, item) for v in self.variables]
        if len(data[0].shape) > 2:
            return np.stack(data, axis=1)
        return np.stack(data, axis=0)

    def _get_variable_item(self, v, item):
        item_data = self.data[v]
        mu, std = item_data.scales
        item_data = (item_data.data.isel(**{self.DUMMY_DIM_NAME: item}) - mu) / std        
        out = item_data.values
        return out


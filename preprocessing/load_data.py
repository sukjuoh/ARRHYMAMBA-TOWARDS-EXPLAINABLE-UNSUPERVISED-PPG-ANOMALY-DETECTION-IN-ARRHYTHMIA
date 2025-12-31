"""
ref: https://github.com/ML4ITS/TimeVQVAE-AnomalyDetection/blob/main/preprocessing/preprocess.py
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from einops import rearrange, repeat
import torch
import numpy as np

from torch.utils.data import Dataset


from utils import set_window_size


data_dir = Path("/preprocessing/dataset/PPG_Dataset")
# print(data_dir)

pattern = re.compile(r"^([0-9]{2})_([a-z]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+).txt$")


@dataclass
class PPG_TestSequence:
    name: str
    id: int

    train_start: int  # starts at 1
    train_stop: int

    anom_start1: int
    anom_stop1: int

    anom_start2: int
    anom_stop2: int

    anom_start3: int
    anom_stop3: int

    data: np.ndarray

    @property
    def train_data(self):

        return self.data[self.train_start : self.train_stop]

    @property
    def test_data(self):

        return self.data[self.train_stop:]

    @property
    def anom_data(self):

        return self.data[self.anom_start1 : self.anom_stop1], self.data[self.anom_start2 : self.anom_stop2], self.data[self.anom_start3 : self.anom_stop3] 


    @classmethod
    def create(cls, path: Path) -> PPG_TestSequence:

        assert path.exists()
        data = np.loadtxt(path, dtype=np.float32)

        match = pattern.match(path.name)
        assert match

        id, name, train_stop, anom_start1, anom_stop1, anom_start2, anom_stop2, anom_start3, anom_stop3 = match.groups()

        return cls(
            name=name,
            id=int(id),
            train_start=1,
            train_stop=int(train_stop) + 1,  # +1 to make python-index easier
            anom_start1=int(anom_start1),
            anom_stop1=int(anom_stop1) + 1,
            anom_start2=int(anom_start2),
            anom_stop2=int(anom_stop2) + 1,
            anom_start3=int(anom_start3),
            anom_stop3=int(anom_stop3) + 1,  
            data=data,
        )

    @classmethod
    def create_by_id(cls, id: int) -> PPG_TestSequence:
        return cls.create((next(data_dir.glob(f"{id:02d}_*"))))

    @classmethod
    def create_by_name(cls, name: str) -> PPG_TestSequence:
        return cls.create((next(data_dir.glob(f"*_{name}_*"))))


class PPGTestDataset(Dataset):
    def __init__(self,
                 kind:str,
                 dataset_importer: PPG_TestSequence,
                 n_period:int,
                 interval:int,
                 ):
        assert kind in ['train', 'test']
        self.kind = kind
        self.n_period = n_period
        self.interval = interval
       
        #to ensure the length of data
        
        data = dataset_importer.train_data
        
        #utilize heartpy library
        self.window_size = set_window_size(data)*self.n_period

        

        if kind == 'train':
            self.X = dataset_importer.train_data[:, None]  # add channel dim; (ts_len, 1)
        elif kind == 'test':
            self.X = dataset_importer.test_data[:, None]  # (ts_len, 1)
            # anomaly data
            self.anom_start1 = dataset_importer.anom_start1 - dataset_importer.train_stop  # relative to `test_data`
            self.anom_stop1 = dataset_importer.anom_stop1 - dataset_importer.train_stop  # relative to `test_data`
            self.anom_start2 = dataset_importer.anom_start2 - dataset_importer.train_stop  # relative to `test_data`
            self.anom_stop2 = dataset_importer.anom_stop2 - dataset_importer.train_stop  # relative to `test_data`
            self.anom_start3 = dataset_importer.anom_start3 - dataset_importer.train_stop  # relative to `test_data`
            self.anom_stop3 = dataset_importer.anom_stop3 - dataset_importer.train_stop  # relative to `test_data`
            self.Y = np.zeros_like(self.X)[:, 0]  # (ts_len,)
            self.Y[self.anom_start1:self.anom_stop1] = 1.
            self.Y[self.anom_start2:self.anom_stop2] = 1.
            self.Y[self.anom_start3:self.anom_stop3] = 1.



        ts_len = self.X.shape[0]
        self.dataset_len = (ts_len - 1) - self.window_size

    def __getitem__(self, idx):
        rng = slice(idx, idx+self.window_size)

        x = self.X[rng]  # (window_size, 1); 1 denotes channel dim.
        x = rearrange(x, 'l c -> c l')  # (1, window_size)
        x = torch.from_numpy(x).float()  # (1, window_size)
        x = scale(x).float()
        
        if self.kind == 'train':
            return x
        elif self.kind == 'test':
            y = self.Y[rng]  # (window_size,)
            y = torch.from_numpy(y).long()  # (window_size,)
            return x, y

    def __len__(self):
        return self.dataset_len

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf
from torchaudio.transforms import Resample


class voiceFilterDataset(Dataset):
    # this assumes preprocessed audios are already there in the paths mentioned in the csv file
    def __init__(self, mode="test", source_sr=16000, target_sr_for_sep=8000, data_store="raid"):
        self.mode = mode
        if data_store == "raid":
            self.csv_path = f"data/{mode}_normalized_mixed.csv"
        elif data_store == "local":
            self.csv_path = f"data/{mode}_normalized_mixed_local.csv"
        else:
            raise ValueError(f"{data_store} data store is not supported. Please choose from 'raid' or 'local'.")
        self.df = pd.read_csv(self.csv_path)
        self.source_sr = source_sr
        self.target_sr = target_sr_for_sep
        self.resampler = Resample(self.source_sr, self.target_sr)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        mix = self.resampler(torch.from_numpy(sf.read(row["mix_path"], dtype="float32")[0]))
        src = self.resampler(torch.from_numpy(sf.read(row["s1_path"], dtype="float32")[0]))
        ref = torch.from_numpy(sf.read(row["r1_path"], dtype="float32")[0])

        return mix, src, ref

    def __len__(self):
        return len(self.df.index)


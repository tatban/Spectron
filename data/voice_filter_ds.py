import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf
from torchaudio.transforms import Resample


class voiceFilterDataset(Dataset):
    # this assumes preprocessed audios are already there in the paths mentioned in the csv file
    def __init__(self, mode="test"):
        self.mode = mode
        self.csv_path = f"data/{mode}_normalized_mixed.csv"
        self.df = pd.read_csv(self.csv_path)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        mix = torch.from_numpy(sf.read(row["mix_path"], dtype="float32")[0])
        src = torch.from_numpy(sf.read(row["s1_path"], dtype="float32")[0])
        ref = torch.from_numpy(sf.read(row["r1_path"], dtype="float32")[0])

        return mix, src, ref

    def __len__(self):
        return len(self.df.index)


# SPI = Speaker Presence Invariant Training
import os

import numpy as np
import pandas as pd
import torch
from asteroid.data import LibriMix
from torch.utils.data import Dataset
import soundfile as sf
from torchaudio.transforms import Resample, MelSpectrogram
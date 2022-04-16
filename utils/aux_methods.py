import glob
import os
import numpy as np
from pathlib import Path
import torch
from scipy.io.wavfile import write


def audio_logger_local(path, mixture, gTs, qurs, sampling_rate=8000, predictions=None, global_step=None, extra_info="", normalize=True):
    """
        This function is designed to work with batch_size=1. To log multiple audios split the batch with size=1 and call this function in a loop
        :param qurs: tensor of shape [1, n_src, audio_sample_length]
        :param path: str (directory path to write the audios)
        :param mixture: tensor of shape [1, audio_sample_length]
        :param gTs: tensor of shape [1, n_src, audio_sample_length]
        :param sampling_rate: int
        :param predictions: tensor of shape [1, n_src, audio_sample_length]
        :param global_step: int (iteration number=epoch*len(dataloader)+bach_index)
        :param extra_info: str (to document current loss value or so in the folder name)
        :param normalize: bool (whether to normalize the audio in -1 to 1 before logging)
        :return: NA
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    # log input and ground truths only once (i.e if doesn't exist)
    if not glob.glob(os.path.join(path, "input_mixture*.wav")):
        # save input mixture
        s = mixture.detach().cpu().numpy().reshape(-1, 1)
        if normalize:
            s = 2. * (s - np.min(s)) / np.ptp(s) - 1
        write(os.path.join(path, "input_mixture.wav"), sampling_rate, s)

    # log ground truths
    if not glob.glob(os.path.join(path, "GT_*.wav")):
        # save gts
        gTs = torch.split(gTs.detach().to('cpu'), 1, 1)
        for i, s in enumerate(gTs):
            s = s.numpy().reshape(-1, 1)
            if normalize:
                s = 2. * (s - np.min(s)) / np.ptp(s) - 1
            write(os.path.join(path, f"GT_S{i}.wav"), sampling_rate, s)

    # log ground queries
    if not glob.glob(os.path.join(path, "QR_*.wav")):
        # save reference speeches
        qurs = torch.split(qurs.detach().to('cpu'), 1, 1)
        for i, s in enumerate(qurs):
            s = s.numpy().reshape(-1, 1)
            if normalize:
                s = 2. * (s - np.min(s)) / np.ptp(s) - 1
            write(os.path.join(path, f"QR_S{i}.wav"), 16000, s)  # sampling rate is hardcoded to 16KHz for reference signals as Spectron uses ref signal at 16KHz

    # save predictions
    if predictions is not None:
        if global_step is not None:
            path = os.path.join(path, str(global_step))
            Path(path).mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(predictions):
            out_audio = s.numpy().reshape(-1, 1)
            if normalize:
                out_audio = 2. * (out_audio - np.min(out_audio)) / np.ptp(out_audio) - 1
            write(os.path.join(path, f"EST_S{i}_{extra_info}.wav"), sampling_rate, out_audio)
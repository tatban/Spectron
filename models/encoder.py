import math
from pathlib import Path
from typing import Union
import torch
import torch.nn as nn
from torchaudio.transforms import Resample, MelSpectrogram
import torch.nn.functional as F
from time import perf_counter as timer


class SECore(nn.Module):  # speaker encoder core
    def __init__(self, mel_n_channels=40, model_hidden_size=256, model_num_layers=3, model_embedding_size=256, device: Union[str, torch.device] = None, verbose=True):
        """
        :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda").
        If None, defaults to cuda if it is available on your machine, otherwise the model will
        run on cpu. Outputs are always returned on the cpu, as numpy arrays.
        """
        super().__init__()

        # Define the network
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Load the pretrained model's weights
        weights_fpath = Path(__file__).resolve().parent.joinpath("pretrained.pt")
        if not weights_fpath.exists():
            raise Exception("Couldn't find the voice encoder pretrained model at %s." %
                            weights_fpath)
        start = timer()
        checkpoint = torch.load(weights_fpath, map_location="cpu")  # todo: make this loading by choice with some arguments
        self.load_state_dict(checkpoint["model_state"], strict=False)
        self.to(device)

        if verbose:
            print("Loaded the voice encoder model on %s in %.2f seconds." %
                  (device.type, timer() - start))

    def forward(self, mels: torch.FloatTensor):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param mels: a batch of mel spectrograms of same duration as a float32 tensor of shape
        (batch_size, n_frames, n_channels)
        :return: the embeddings as a float 32 tensor of shape (batch_size, embedding_size).
        Embeddings are positive and L2-normed, thus they lay in the range [0, 1].
        """
        # Pass the input through the LSTM layers and retrieve the final hidden state of the last
        # layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)


class SpeakerEncoder(nn.Module):
    def __init__(self, source_sr, sampling_rate=16000, mel_window_length=25, mel_window_step=10, mel_n_channels=40, partials_n_frames=160, audio_norm_target_dBFS=-30, device="cuda", fixed_encoder=True, min_coverage=0.75, rate=1.3):
        """
        Computes representative speaker embedding vector of 256 length from clean speech audio.
        This embedding is used later to condition the separation module.
        :param source_sr: sampling rate of the input
        :param sampling_rate: target sample rate (this has implication on encoder model weights dimension)
        :param mel_window_length:
        :param mel_window_step:
        :param mel_n_channels:
        :param partials_n_frames:
        :param audio_norm_target_dBFS: target volume for normalization
        :param device: CUDA or CPU or so
        :param fixed_encoder: if set, uses a pre-trained fixed speaker encoder (from GE2E loss)
        :param rate: how many partial utterances should occur per second. Partial utterances must
        cover the span of the entire utterance, thus the rate should not be lower than the inverse
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
        then the last partial utterance will be considered by zero-padding the audio. Otherwise,
        it will be discarded. If there aren't enough frames for one partial utterance,
        this parameter is ignored so that the function always returns at least one slice.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        super(SpeakerEncoder, self).__init__()
        self.device = device

        # region hypeparams
        self.source_sr = source_sr
        self.sampling_rate = sampling_rate
        self.mel_window_length = mel_window_length
        self.mel_window_step = mel_window_step
        self.mel_n_channels = mel_n_channels
        self.partials_n_frames = partials_n_frames

        # volume normalize hyper params
        self.norm_vol = True
        self.int16_max = (2 ** 15) - 1
        self.target_dBFS = audio_norm_target_dBFS
        self.increase_only = True
        self.decrease_only = False

        # partial slice hyperparams
        self.min_coverage = min_coverage
        self.rate = rate
        # endregion

        # region transformations
        self.resampler = Resample(source_sr, sampling_rate)
        self.melspect = MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=int(sampling_rate * mel_window_length / 1000),
            hop_length=int(sampling_rate * mel_window_step / 1000),
            n_mels=mel_n_channels
        )
        # endregion

        # core encoder from mel to embedding
        self.core = SECore(device=self.device)
        if fixed_encoder:  # freeze the speaker encoder
            for p in self.core.parameters():
                p.requires_grad = False

    def normalize_volume(self, wav):
        if self.increase_only and self.decrease_only:
            raise ValueError("Both increase only and decrease only can't be set simultaneously")
        rms = torch.sqrt(torch.mean((wav * self.int16_max) ** 2, dim=-1, keepdim=True))  # shape: N x n_src x 1
        wave_dBFS = 20 * torch.log10(rms / self.int16_max)  # shape: N x n_src x 1
        dBFS_change = self.target_dBFS - wave_dBFS  # shape: N x n_src x 1
        if self.increase_only:
            dBFS_change = F.relu(dBFS_change)  # replace all negative changes with zero
        elif self.decrease_only:
            dBFS_change = -F.relu(-dBFS_change)  # replace all positive changes with zero by double negation technique
        return wav * (10 ** (dBFS_change / 20))  # shape: N x n_src x time (same as wav)

    def compute_partial_slices(self, n_samples):  # n_samples is an int
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to
        obtain partial utterances of <partials_n_frames> each. Both the waveform and the
        mel spectrogram slices are returned, so as to make each partial utterance waveform
        correspond to its spectrogram.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wav_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        """
        assert 0 < self.min_coverage <= 1

        # Compute how many frames separate two partial utterances
        samples_per_frame = int((self.sampling_rate * self.mel_window_step / 1000))
        n_frames = int(math.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(round((self.sampling_rate / self.rate) / samples_per_frame))
        assert 0 < frame_step, "The rate is too high"
        assert frame_step <= self.partials_n_frames, "The rate is too low, it should be %f at least" % \
                                                     (self.sampling_rate / (samples_per_frame * self.partials_n_frames))

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - self.partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = torch.tensor([i, i + self.partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < self.min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def forward(self, wav_batch):  # wav_batch.shape = (N, n_src=1, time(or more specifically Number of Samples))
        if wav_batch.ndim == 2:
            wav_batch = wav_batch.unsqueeze(1)
        # resample
        if self.source_sr != self.sampling_rate:
            wav_batch = self.resampler(wav_batch)
        # normalize volume
        if self.norm_vol:
            wav_batch = self.normalize_volume(wav_batch)
        # compute slices
        wav_slices, mel_slices = self.compute_partial_slices(wav_batch.size(-1))  # last dimension is assumed to be the time (audio length) dimension, so it says how many total samples are there
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= wav_batch.size(-1):
            wav_batch = F.pad(wav_batch, (0, max_wave_length - wav_batch.size(-1)), "constant")
        # MelSpectogram
        mel_batch = self.melspect(wav_batch)  # output shape: N x n_src=1 x mel_n_channels x n_frames
        mel_b = mel_batch.permute(1, 0, 3, 2)
        partial_embeds = []
        # Mel slices
        for s in mel_slices:  # iterate over partial slices
            # pass through encoder core
            partial_embeds.append(self.core(mel_b.squeeze(0)[:, s, :]))
        # Average and normalize
        aggr_embedding = torch.stack(partial_embeds, dim=0)  # op dim: np x N x embed_size where np = number of partial slices
        centroid_embedding = torch.mean(aggr_embedding, dim=0, keepdim=False)  # op dim: N x embed_size
        centroid_embedding = centroid_embedding / torch.linalg.norm(centroid_embedding, ord=2, dim=-1, keepdim=True)  # op dim: N x embed_size

        return centroid_embedding  # op dim: N x embed_size  where N is batch size

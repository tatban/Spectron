import os
import statistics
import sys
from torch.utils.data import DataLoader
from pathlib import Path
from data import voiceFilterDataset as VFDS
import torch
from models import Spectron, SpeakerEncoder
from asteroid.losses.sdr import SingleSrcNegSDR
import logging
from mir_eval.separation import bss_eval_sources

OUT_DIR = "/mnt/raid/tbandyo/idp4vc_ws/QSEP_LOGS/inference_eval"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.DEBUG, filename=os.path.join(OUT_DIR, "inference_logs.txt"), filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


si_snr_eval = SingleSrcNegSDR(sdr_type="sisdr")


def inference_batch(sep_model, enc_model, test_loader, device, exp_nm, ds_nm):
    sep_model.eval()
    enc_model.eval()
    sep_model.to(device)
    enc_model.to(device)

    SDRs = []
    SDRis = []
    SI_SNRs = []
    SI_SNRis = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            mix, src, qrs = batch

            mix = mix.to(device)
            src = src.to(device)
            qrs = qrs.to(device)

            # fwd pass
            # iterate over the number of speakers (sources)
            spk_embed = enc_model(qrs)
            est_speech, _ = sep_model(mix, spk_embed)  # dim: N x T where N = batch size
            target_speech = src  # dim: N x T where N = batch size

            # compute SDR and SDRi
            for i in range(src.shape[0]):  # intra batch looping as bss_eval doesn't support batched data
                sdr, _, _, _ = bss_eval_sources(
                    target_speech[i, :].detach().cpu().numpy(),
                    est_speech[i, :].detach().cpu().numpy(),
                )
                SDRs.append(sdr.mean())
                sdr_baseline, _, _, _ = bss_eval_sources(
                    target_speech[i, :].detach().cpu().numpy(),
                    mix[i, :].detach().cpu().numpy(),
                )
                sdr_i = sdr.mean() - sdr_baseline.mean()
                SDRis.append(sdr_i)

            # compute SI-SNR (also known as SI-SDR) and SI-SNRi
            si_snr = -si_snr_eval(est_speech.squeeze(), target_speech.squeeze())  # dim: (N,) where N = batch size
            si_snr_baseline = -si_snr_eval(mix.squeeze(), target_speech.squeeze())
            si_snri = si_snr - si_snr_baseline  # dim: (N,) where N = batch size
            SI_SNRs.extend(si_snr.squeeze().cpu().tolist())
            SI_SNRis.extend(si_snri.squeeze().cpu().tolist())
        logging.info(f"{exp_nm}_{ds_nm}\t\tAvg SDR: {statistics.mean(SDRs):.4f}\t\tAvg SDRi: {statistics.mean(SDRis):.4f}\t\tAvg SI-SNR: {statistics.mean(SI_SNRs):.4f}\t\tAvg SI-SNRi: {statistics.mean(SI_SNRis):.4f}")


if __name__ == "__main__":
    # settings for DPTNet
    FF_HID = 256
    FF_ACTIVATION = "relu"
    CHUNK_SIZE = 100
    HOP_SIZE = None
    N_REPEATS = 6
    NORM_TYPE = "gLN"
    MASK_ACT = "relu"
    BIDIRECTIONAL = True
    DROPOUT = 0
    N_FILTERS = 64
    ENCODER_ACT = "relu"

    SOURCE_SR = 16000
    SAMPLE_RATE = 8000
    DEVICE = "cuda"
    FIXED_SE = False
    BACK_BONE = "DPTNet"
    N_SRC = 2
    BATCH_SIZE = 16
    N_WORKERS = 16

    exp_name = "Spectron"
    dataset = "VFDS"
    weights_path = "/mnt/raid/tbandyo/idp4vc_ws/SPECTRON_LOGS/VOICE_FILTER_DS/best_model_for_voice_filter_dataset.pth"

    test_set = VFDS(mode="test")
    test_ldr = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

    speaker_encoder = SpeakerEncoder(source_sr=SOURCE_SR, device=DEVICE, fixed_encoder=FIXED_SE).to(DEVICE)
    model = Spectron().to(DEVICE)

    state = torch.load(weights_path)
    model.load_state_dict(state["SPECTRON"])
    speaker_encoder.load_state_dict(state["SE"])
    for p in model.parameters():
        p.requires_grad = False
    for pse in speaker_encoder.parameters():
        pse.requires_grad = False

    speaker_encoder.eval()
    model.eval()

    inference_batch(model, speaker_encoder, test_ldr, DEVICE, exp_name, dataset)




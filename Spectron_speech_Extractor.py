import os
import random
import numpy as np
import torch
from models import Spectron, SpeakerEncoder
from pathlib import Path
import soundfile as sf
from scipy.io.wavfile import write
from torchaudio.transforms import Resample

SOURCE_SR_ref = 16000
SOURCE_SR_mix = 8000
DEVICE = "cuda"
FIXED_SE = False
SEED = 3407


# set random seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
generator = torch.Generator().manual_seed(SEED)


def extract_speeches(root_path=r"C:\Users\Tathagata\OneDrive\Desktop\Spectron_results\wildmix", model_weights_path=r"C:\Users\Tathagata\OneDrive\Desktop\Spectron_results\best_spectron_model.pth"):
    """
    This method assumes a folder structure as follows:
        root_path
            |--> mix
                  |--> <something>_mix_1.wav
                  |--> <something>_mix_2.wav
                  |--> ...
            |--> ref
                  |--> <something>_ref_1.wav
                  |--> <something>_ref_2.wav
                  |--> ...
    :param root_path: path of the root folder where mix and ref folders are there
    :param model_weights_path: dictionary containing the model weights
    :return: True on successful saving on extracted speeches in est folder under root folder
    """
    if model_weights_path is None:
        raise ValueError("model weights can't be none")

    # load models
    spk_enc = SpeakerEncoder(source_sr=SOURCE_SR_ref, device=DEVICE, fixed_encoder=FIXED_SE).to(DEVICE)
    sep_model = Spectron().to(DEVICE)
    model_state = torch.load(model_weights_path)
    spk_enc.load_state_dict(model_state['SE'])
    sep_model.load_state_dict(model_state['SPECTRON'])
    spk_enc.eval()
    sep_model.eval()

    mixture_path = os.path.join(root_path, "mix")
    reference_path = os.path.join(root_path, "ref")
    est_path = os.path.join(root_path, "est")
    Path(est_path).mkdir(parents=True, exist_ok=True)

    # process audios in loop:
    mix_files = Path(mixture_path).glob('*.wav')
    for file in mix_files:
        method_name, _, file_number = file.name.split("_")

        mixed_sample, msr = sf.read(file, dtype="float32")
        reference_sample, rsr = sf.read(os.path.join(reference_path, "_".join([method_name, "ref", file_number])), dtype="float32")

        mix_w = torch.from_numpy(mixed_sample).unsqueeze(0).to(DEVICE)
        query_w = torch.from_numpy(reference_sample).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if msr != SOURCE_SR_mix:
                mix_w = Resample(msr, SOURCE_SR_mix).to(DEVICE)(mix_w)
            if rsr != SOURCE_SR_ref:
                query_w = Resample(rsr, SOURCE_SR_ref).to(DEVICE)(query_w)

            speaker_embeds = spk_enc(query_w)
            est_speech, _ = sep_model(mix_w, speaker_embeds)
            prediction = est_speech.detach().squeeze().to('cpu')
        out_audio = prediction.numpy().reshape(-1, 1)  # Warning!!!: works only for batch size 1 and mono audios
        out_audio = 2. * (out_audio - np.min(out_audio)) / np.ptp(out_audio) - 1
        write(os.path.join(est_path, f"spectron_out_{file_number}"), SOURCE_SR_mix, out_audio)


if __name__ == "__main__":
    extract_speeches()




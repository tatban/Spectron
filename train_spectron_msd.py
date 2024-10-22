import os
import random
import sys
import numpy as np
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from data import voiceFilterDataset as VFDS
import torch
from models import Spectron, SpeakerEncoder
from models.adversarial_models import MultiScaleDiscriminator, discriminator_loss, generator_loss
from asteroid.losses.sdr import SingleSrcNegSDR
import logging
from utils import audio_logger_local


# arguments / hyper parameters:
DEVICE = "cuda"
SEED = 3407
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-7
MAX_EPOCHS = 201
SDR_TYPE = 'sisdr'
WFRL_WEIGHT = 1
SECL_WEIGHT = 1e3
TFCL_WEIGHT = 1e6
BATCH_PRINT_STRIDE = 1000
VAL_BATCH_PRINT_STRIDE = 89
SOURCE_SR = 16000  # only for speaker encoder, for separator SR=8000
FIXED_SE = False
OUT_DIR = "/mnt/raid/tbandyo/idp4vc_ws/SPECTRON_GAN_LOGS/VOICE_FILTER_DS/SPECTRON_MSD"
FIXED_AUDIO_INDEX = [1588, 2335, 1607, 203, 308, 1592, 1095, 2216, 2187, 1790, 2191, 1596]
NORMALIZE_AUDIO_BEFORE_SAVE = True


# logging
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.DEBUG, filename=os.path.join(OUT_DIR, "log.txt"), filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
writer = SummaryWriter(log_dir=OUT_DIR)


# set random seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
generator = torch.Generator().manual_seed(SEED)  # used for dataset splitting
torch.backends.cudnn.benchmark = False

# autograd debug
torch.autograd.set_detect_anomaly(True)

# data
dataset = VFDS(mode="train")
train_size = int(len(dataset) * 0.95)  # 95 percent of the loaded dataset will be used as training dataset
val_size = len(dataset) - train_size  # rest 5 percent will be used as validation set
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# models
spk_enc = SpeakerEncoder(source_sr=SOURCE_SR, device=DEVICE, fixed_encoder=FIXED_SE).to(DEVICE)
sep_model = Spectron().to(DEVICE)
msd = MultiScaleDiscriminator().to(DEVICE)

# objective functions
re_criterion = SingleSrcNegSDR(sdr_type=SDR_TYPE, reduction="mean").to(DEVICE)  # waveform reconstruction loss
em_criterion = MSELoss().to(DEVICE)  # for speaker embedding consistency loss
tf_criterion = MSELoss().to(DEVICE)  # for time-frequency encoder decoder consistency

# optimizer
optimizer = Adam(list(sep_model.parameters()) + list(spk_enc.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optim_d = torch.optim.AdamW(msd.parameters(), 0.0002, betas=(0.8, 0.99))  # discriminator optimizer

# misc.
train_loader_length = len(train_loader)
val_loader_length = len(val_loader)
best_val_loss = 1000
best_epoch = -1


# loop
if __name__ == "__main__":
    for e in range(MAX_EPOCHS):
        logging.info(f"\nRunning epoch: {e + 1} of {MAX_EPOCHS}\n------------------------\n")

        # training
        spk_enc.train()
        sep_model.train()
        msd.train()
        logging.info("Training Loop:")
        for train_batch_idx, train_batch in enumerate(train_loader):
            step = e * train_loader_length + train_batch_idx
            optimizer.zero_grad()

            mix, src, qrs = train_batch
            mix = mix.to(DEVICE)  # dim: N x T(no of temporal samples)
            src = src.to(DEVICE)  # dim: N x T(no of temporal samples)
            qrs = qrs.to(DEVICE)  # dim: N x T(no of temporal samples)

            # get reference speech embedding
            speaker_embeds = spk_enc(qrs)  # speaker_embeds refers to the mean / centroid embedding of the reference speech, dim: N x embed_length

            # get the estimated clean speech from model forward
            est_speech, tf_op = sep_model(mix, speaker_embeds)   # est_speech dim: N x 1 x T , tf_op dim: N x 1 x W x H

            # get the consistency outputs
            # embedding consistency
            est_speaker_embeds = spk_enc(est_speech)  # est_speaker_embeds refers to embedding obtained from the estimated clean speech, dim: N x embed_length

            # TF consistency
            est_tf_rep = sep_model.forward_encoder(est_speech)  # dim: N x W x H

            # calculate training losses
            wfrl = re_criterion(est_speech.squeeze(1), src.squeeze(1))  # Wave Form Reconstruction quality Loss
            secl = em_criterion(est_speaker_embeds, speaker_embeds)  # Speaker Embedding Consistency Loss
            tfcl = tf_criterion(est_tf_rep, tf_op.squeeze(1))  # Time Frequency Consistency Loss

            # multi scale discriminator (MSD)
            y = torch.autograd.Variable(src.unsqueeze(1))  # real signal
            optim_d.zero_grad()

            # MSD calculation
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, est_speech.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r,
                                                                               y_ds_hat_g)
            loss_disc_s.backward()
            optim_d.step()

            # generator
            optimizer.zero_grad()
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, est_speech)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            total_loss = loss_gen_s + (WFRL_WEIGHT * wfrl) + (SECL_WEIGHT * secl) + (TFCL_WEIGHT * tfcl)

            # backprop
            total_loss.backward()

            # update weights
            optimizer.step()  # might need to update the weights inside the src inner loop

            # print / log losses
            writer.add_scalar("Training Loss/WFRL", wfrl.item() * WFRL_WEIGHT, global_step=step)
            writer.add_scalar("Training Loss/SECL", secl.item() * SECL_WEIGHT, global_step=step)
            writer.add_scalar("Training Loss/TFCL", tfcl.item() * TFCL_WEIGHT, global_step=step)
            writer.add_scalar("Training Loss/ad_gen_batch", loss_gen_s.item(), global_step=step)
            writer.add_scalar("Training Loss/ad_disc_batch", loss_disc_s.item(), global_step=step)
            writer.add_scalar("Training Loss/Total Loss", total_loss.item(), global_step=step)
            if (train_batch_idx + 1) % BATCH_PRINT_STRIDE == 0:
                logging.info(f"Batch: {train_batch_idx + 1}\t\tWFRL: {wfrl.item():.4f}\t\tSECL: {secl.item() * SECL_WEIGHT:.4f}\t\tTFCL: {tfcl.item() * TFCL_WEIGHT:.4f}\t\tad_gen_batch: {loss_gen_s.item():.4f}\t\tad_disc_batch: {loss_disc_s.item():.4f}\t\tTOTAL: {total_loss.item():.4f}")

        # validation
        spk_enc.eval()
        sep_model.eval()
        logging.info("\nValidation Loop:")
        val_losses = []
        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_loader):
                step = e * val_loader_length + val_batch_idx

                mix_v, src_v, qrs_v = val_batch
                mix_v = mix_v.to(DEVICE)
                src_v = src_v.to(DEVICE)
                qrs_v = qrs_v.to(DEVICE)

                # get reference speech embedding
                speaker_embeds_v = spk_enc(qrs_v)  # speaker_embeds refers to the mean / centroid embedding of the reference speech, dim: N x embed_length

                # get the estimated clean speech from model forward
                est_speech_v, tf_op_v = sep_model(mix_v, speaker_embeds_v)  # est_speech dim: N x 1 x T , tf_op dim: N x W x H

                # get the consistency outputs
                # embedding consistency
                est_speaker_embeds_v = spk_enc(est_speech_v)  # est_speaker_embeds refers to embedding obtained from the estimated clean speech, dim: N x embed_length

                # TF consistency
                est_tf_rep_v = sep_model.forward_encoder(est_speech_v)  # dim: N x W x H

                # calculate training losses
                wfrl_v = re_criterion(est_speech_v.squeeze(1), src_v.squeeze(1))  # Wave Form Reconstruction quality Loss
                secl_v = em_criterion(est_speaker_embeds_v, speaker_embeds_v)  # Speaker Embedding Consistency Loss
                tfcl_v = tf_criterion(est_tf_rep_v, tf_op_v.squeeze(1))  # Time Frequency Consistency Loss

                total_loss_v = (WFRL_WEIGHT * wfrl_v) + (SECL_WEIGHT * secl_v) + (TFCL_WEIGHT * tfcl_v)

                val_losses.append(total_loss_v.item())

                writer.add_scalar("Validation Loss/WFRL", wfrl_v.item() * WFRL_WEIGHT, global_step=step)
                writer.add_scalar("Validation Loss/SECL", secl_v.item() * SECL_WEIGHT, global_step=step)
                writer.add_scalar("Validation Loss/TFCL", tfcl_v.item() * TFCL_WEIGHT, global_step=step)
                writer.add_scalar("Validation Loss/Total Loss", total_loss_v.item(), global_step=step)
                if (val_batch_idx + 1) % VAL_BATCH_PRINT_STRIDE == 0:
                    logging.info(f"Batch: {val_batch_idx + 1}\t\tWFRL: {wfrl_v.item():.4f}\t\tSECL: {secl_v.item() * SECL_WEIGHT:.4f}\t\tTFCL: {tfcl_v.item() * TFCL_WEIGHT:.4f}\t\tTOTAL: {total_loss_v.item():.4f}")

            val_loss_epoch = sum(val_losses) / len(val_losses)
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_epoch = e
                # log model
                torch.save(
                    {
                        'SPECTRON': sep_model.state_dict(),  # separation model weights
                        'SE': spk_enc.state_dict()  # speaker encoder weights
                    },
                    os.path.join(OUT_DIR, f"best_model_for_voice_filter_dataset.pth")
                )

                # log audio samples
                for idx in FIXED_AUDIO_INDEX:
                    if idx < len(val_dataset):
                        mx_f, gt_f, qr_f = val_dataset[idx]
                        mx_f = mx_f.unsqueeze(0).to(DEVICE)
                        gt_f = gt_f.unsqueeze(0).unsqueeze(0)
                        qr_f = qr_f.unsqueeze(0).to(DEVICE)

                        se_f = spk_enc(qr_f)
                        op_f, _ = sep_model(mx_f, se_f)

                        audio_logger_local(os.path.join(OUT_DIR, "Audio Samples", f"index_{idx}"), mx_f, gt_f, qr_f.unsqueeze(0), predictions=[op_f.detach().to('cpu')], global_step=e, extra_info=f"avg_loss_{val_loss_epoch}", normalize=NORMALIZE_AUDIO_BEFORE_SAVE)

    logging.info(f"\nTraining complete. Best epoch validation loss is {best_val_loss} achieved in epoch {best_epoch}.")

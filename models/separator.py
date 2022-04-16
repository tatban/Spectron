import torch
import torch.nn as nn
from asteroid.models.base_models import BaseEncoderMaskerDecoder, _shape_reconstructed, _unsqueeze_to_3d
from asteroid_filterbanks import make_enc_dec
from asteroid.masknn import TDConvNet, DPTransformer
from asteroid.utils.torch_utils import jitable_shape, pad_x_to_y


class Spectron(BaseEncoderMaskerDecoder):
    """DPTNet separation model, as described in [1]. Modification: speaker conditioned masking

    Args:
        n_src (int): Number of sources in the input mixtures. Should always be one for modified formulation.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        causal (bool, optional) : Whether or not the convolutions are causal.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1] : "Dual-Path Transformer Network: Direct Context-Aware Modeling for
                End-to-End Monaural Speech Separation" INTERSPEECH 2020, Jingjing Chen,
                Qirong Mao, Dong Liu https://arxiv.org/abs/1809.07454
    """

    def __init__(
            self,
            n_src_to_est_at_once=1,
            out_chan=None,  # only for conv_tasnet
            n_blocks=8,  # only for conv_tasnet
            n_repeats=6,  # 6 for DPTNet, 3 for conv_tasnet, for DPTNet this refers to how many transformer layers will be used
            bn_chan=128,  # only for conv_tasnet
            hid_chan=512,  # only for conv_tasnet
            speaker_embedding_dim=256,
            separate_conditioner=True,
            skip_chan=128,  # only for conv_tasnet
            conv_kernel_size=3,  # only for conv_tasnet
            norm_type="gLN",
            mask_act="relu",  # "sigmoid",  # mask output activation
            in_chan=None,
            causal=False,  # only for conv_tasnet
            fb_name="free",
            kernel_size=16,  # filter kernel size of adaptive front tf encoder and decoder/vocoder
            n_filters=64,  # 64 for DPTNet, 512 for conv_tasnet, it refers to number of filters used in adaptive tf encoder and also decoder/vocoder
            stride=8,  # stride size of adaptive front tf encoder and decoder/vocoder
            encoder_activation="relu",  # None,  # activation to be applied after adaptive front tf encoder block, default identity or no activation function
            sample_rate=8000,
            back_bone="DPTNet",  # "conv_tasnet"
            n_heads=8,  # number of attention heads in the transformer blocks  # only for DPTNet
            ff_hid=256,  # 1024,  # 256,  # only for DPTNet, hidden dimension of DPTNet layer (no of neurones in the hidden layer of the single hidden layer fully connected MLP inside transformer block)
            chunk_size=100,  # 250,  # temporal length of a chunk (after chunk splitting of the conditioned tf encoder output) that is going to be processed by intra-transformer   # only for DPTNet
            hop_size=None,  # only for DPTNet
            ff_activation="relu",  # activation to be used in side the transformer layers  # only for DPTNet
            dropout=0,  # only for DPTNet
            bidirectional=True,
            **fb_kwargs,
    ):
        assert n_src_to_est_at_once == 1, "n_src_to_est_at_once must be 1 for conditional speaker separation"
        self.back_bone = back_bone
        self.sep_conditioner = separate_conditioner
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )

        if self.back_bone == "conv_tasnet":  # only for ablation
            if self.sep_conditioner:  # if use a different conv layer to merge the speaker embedding (condition)
                masker = TDConvNet(
                    n_feats,
                    n_src_to_est_at_once,
                    out_chan=out_chan,
                    n_blocks=n_blocks,
                    n_repeats=n_repeats,
                    bn_chan=bn_chan,
                    hid_chan=hid_chan,
                    skip_chan=skip_chan,
                    conv_kernel_size=conv_kernel_size,
                    norm_type=norm_type,
                    mask_act=mask_act,
                    causal=causal,
                )
            else:  # if directly concatenate speaker embedding to transformed speech input representation
                out_chan = out_chan if out_chan else n_feats
                masker = TDConvNet(
                    n_feats + speaker_embedding_dim,
                    n_src_to_est_at_once,
                    out_chan=out_chan,
                    n_blocks=n_blocks,
                    n_repeats=n_repeats,
                    bn_chan=bn_chan,
                    hid_chan=hid_chan,
                    skip_chan=skip_chan,
                    conv_kernel_size=conv_kernel_size,
                    norm_type=norm_type,
                    mask_act=mask_act,
                    causal=causal,
                )
        elif self.back_bone == "DPTNet":  # default for spectron, always uses a separate conditioner layer to merge speaker embedding with transformed speech representation (similar to time frequency representation)
            masker = DPTransformer(
                n_feats,
                n_src_to_est_at_once,
                n_heads=n_heads,
                ff_hid=ff_hid,
                ff_activation=ff_activation,
                chunk_size=chunk_size,
                hop_size=hop_size,
                n_repeats=n_repeats,
                norm_type=norm_type,
                mask_act=mask_act,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        else:
            raise ValueError("Chosen backbone is not supported. Please choose either 'DPTNet' (for default spectron) or 'conv_tasnet' (for ablations)")
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)
        self.conditioner = nn.Conv1d(n_feats + speaker_embedding_dim, n_feats, 1) if self.sep_conditioner else nn.Identity()

    def forward(self, wav, speaker_embedding):  # violets LSP principle as no *args and **kwargs are there in base class forward method. todo: Think of a better design maybe
        # speaker_embedding is assumed to be of shape N x embedding_dimension, where N is batch size
        # wav is assumed to be N x T(number of temporal samples), where N is batch size

        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)

        # get learned spectrogram like time frequency representation
        tf_rep = self.forward_encoder(wav)

        # apply speaker condition (embedding) on the transformed input representation
        tf_len = tf_rep.shape[-1]  # may break or not work properly for stereo audio
        conditioned_tf = torch.cat((tf_rep, speaker_embedding.unsqueeze(-1).expand(-1, -1, tf_len)), dim=1)  # conditioned_tf dim: (N*n_src) x (n_feats+speaker_embedding_dimension) x tf_len
        conditioner_op = self.conditioner(conditioned_tf)

        # compute separation mask
        est_masks = self.forward_masker(conditioner_op)  # est_masks dim: (N*n_src) x 1 x W x H

        # apply mask in the transformed input space
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)

        # compute the wave form of the estimated clean speech
        decoded = self.forward_decoder(masked_tf_rep)  # dim: (N) x 1 x time (n_samples)

        reconstructed = pad_x_to_y(decoded, wav)
        estimated_wav = _shape_reconstructed(reconstructed, shape)
        return estimated_wav, masked_tf_rep  # masked_tf_rep is later used for TF encoder decoder consistency (TFCL)

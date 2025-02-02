#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Self-attention encoder for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import shutil
import torch.nn as nn

from neural_sp.models.modules.linear import Linear
from neural_sp.models.modules.transformer import PositionalEncoding
from neural_sp.models.modules.transformer import TransformerEncoderBlock
from neural_sp.models.seq2seq.encoders.conv import ConvEncoder
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase
from neural_sp.models.torch_utils import make_pad_mask
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')


class TransformerEncoder(EncoderBase):
    """Transformer encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        attn_type (str): type of attention
        attn_n_heads (int): number of heads for multi-head attention
        n_layers (int): number of blocks
        d_model (int): dimension of keys/values/queries in
            MultiheadAttentionMechanism, also the input size of
            the first-layer of the PositionwiseFeedForward
        d_ff (int): dimension of the second layer of the PositionwiseFeedForward
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        last_proj_dim (int): dimension of the last projection layer
        n_stacks (int): number of frames to stack
        n_splices (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channles in the CNN blocks
        conv_kernel_sizes (list): size of kernels in the CNN blocks
        conv_strides (list): number of strides in the CNN blocks
        conv_poolings (list): size of poolings in the CNN blocks
        conv_batch_norm (bool): apply batch normalization only in the CNN blocks
        conv_residual (bool): add residual connection between each CNN block
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and self-attention layers
        param_init (float): only for CNN layers before Transformer layers

    """

    def __init__(self,
                 input_dim,
                 attn_type,
                 attn_n_heads,
                 n_layers,
                 d_model,
                 d_ff,
                 pe_type='add',
                 layer_norm_eps=1e-12,
                 dropout_in=0,
                 dropout=0,
                 dropout_att=0,
                 last_proj_dim=0,
                 n_stacks=1,
                 n_splices=1,
                 conv_in_channel=1,
                 conv_channels=0,
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 conv_poolings=[],
                 conv_batch_norm=False,
                 conv_residual=False,
                 conv_bottleneck_dim=0,
                 param_init=0.1):

        super(TransformerEncoder, self).__init__()
        logger = logging.getLogger("training")

        self.d_model = d_model
        self.n_layers = n_layers
        self.attn_n_heads = attn_n_heads
        self.pe_type = pe_type

        # Setting for CNNs before RNNs
        if conv_channels:
            channels = [int(c) for c in conv_channels.split('_')] if len(conv_channels) > 0 else []
            kernel_sizes = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                            for c in conv_kernel_sizes.split('_')] if len(conv_kernel_sizes) > 0 else []
            strides = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                       for c in conv_strides.split('_')] if len(conv_strides) > 0 else []
            poolings = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))]
                        for c in conv_poolings.split('_')] if len(conv_poolings) > 0 else []
        else:
            channels = []
            kernel_sizes = []
            strides = []
            poolings = []
            logger.warning('Subsampling is automatically ignored because CNN layers are used before RNN layers.')

        if len(channels) > 0:
            assert n_stacks == 1 and n_splices == 1
            self.conv = ConvEncoder(input_dim,
                                    in_channel=conv_in_channel,
                                    channels=channels,
                                    kernel_sizes=kernel_sizes,
                                    strides=strides,
                                    poolings=poolings,
                                    dropout=0,
                                    batch_norm=conv_batch_norm,
                                    residual=conv_residual,
                                    bottleneck_dim=d_model,
                                    param_init=param_init)
            self._output_dim = self.conv.output_dim
        else:
            self._output_dim = input_dim * n_splices * n_stacks
            self.conv = None

            self.embed = Linear(self._output_dim, d_model)  # NOTE: do not apply dropout here

        self.pos_enc = PositionalEncoding(d_model, dropout_in, pe_type)
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, d_ff, attn_type, attn_n_heads,
                                     dropout, dropout_att, layer_norm_eps) for l in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)

        if last_proj_dim != self.output_dim:
            self.bridge = Linear(self._output_dim, last_proj_dim)
            self._output_dim = last_proj_dim
        else:
            self.bridge = None
            self._output_dim = d_model

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with xavier_uniform style."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if 'conv' in n:
                continue  # for CNN layers before Transformer layers
            if p.dim() == 1:
                nn.init.constant_(p, val=0)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0))
            elif p.dim() == 2:
                if 'embed' in n:
                    nn.init.normal_(p, mean=0, std=1 / math.sqrt(self.d_model))
                    logger.info('Initialize %s with %s / %.3f' % (n, 'normal', 1 / math.sqrt(self.d_model)))
                else:
                    nn.init.xavier_uniform_(p, gain=1.0)
                    logger.info('Initialize %s with %s' % (n, 'xavier_uniform'))
            else:
                raise ValueError

    def forward(self, xs, xlens, task):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): `[B]`
            task (str): not supported now
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T, d_model]`
                xlens (list): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None},
                 'ys_sub1': {'xs': None, 'xlens': None},
                 'ys_sub2': {'xs': None, 'xlens': None}}

        if self.conv is None:
            xs = self.embed(xs)
        else:
            # Path through CNN blocks before RNN layers
            xs, xlens = self.conv(xs, xlens)

        # Create the self-attention mask
        bs, xmax = xs.size()[: 2]
        xx_mask = make_pad_mask(xlens, self.device_id).unsqueeze(1).expand(bs, xmax, xmax)
        xx_mask = xx_mask.unsqueeze(1).expand(bs, self.attn_n_heads, xmax, xmax)

        xs = self.pos_enc(xs)
        for l in range(self.n_layers):
            xs, xx_aws = self.layers[l](xs, xx_mask)
            if not self.training:
                setattr(self, 'xx_aws_layer%d' % l, tensor2np(xx_aws))
        xs = self.norm_out(xs)

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        eouts['ys']['xs'] = xs
        eouts['ys']['xlens'] = xlens
        return eouts

    def _plot_attention(self, save_path, n_cols=2):
        """Plot attention for each head in all layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        save_path = mkdir_join(save_path, 'enc_xx_att_weights')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        for l in range(self.n_layers):
            if not hasattr(self, 'xx_aws_layer%d' % l):
                continue

            xx_aws = getattr(self, 'xx_aws_layer%d' % l)

            plt.clf()
            fig, axes = plt.subplots(self.attn_n_heads // n_cols, n_cols, figsize=(20, 8))
            for h in range(self.attn_n_heads):
                if self.attn_n_heads > n_cols:
                    ax = axes[h // n_cols, h % n_cols]
                else:
                    ax = axes[h]
                ax.imshow(xx_aws[-1, h, :, :], aspect="auto")
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'layer%d.png' % (l)), dvi=500)
            plt.close()

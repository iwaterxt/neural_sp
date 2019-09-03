#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN decoder (including CTC loss calculation)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.criterion import distillation
from neural_sp.models.criterion import focal_loss
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.modules.embedding import Embedding
from neural_sp.models.modules.linear import Linear
from neural_sp.models.modules.mocha import MoChA
from neural_sp.models.modules.multihead_attention import MultiheadAttentionMechanism
from neural_sp.models.modules.singlehead_attention import AttentionMechanism
from neural_sp.models.seq2seq.decoders.ctc import CTC
from neural_sp.models.seq2seq.decoders.ctc import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import make_pad_mask
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join
import GPUtil

import matplotlib
matplotlib.use('Agg')

random.seed(1)


class RNNDecoder(DecoderBase):
    """RNN decoder.

    Args:
        eos (int): index for <eos> (shared with <sos>)
        unk (int): index for <unk>
        pad (int): index for <pad>
        blank (int): index for <blank>
        enc_n_units (int): number of units of the encoder outputs
        attn_type (str): type of attention mechanism
        rnn_type (str): lstm/gru
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        residual (bool): add residual connections between each layer
        bottleneck_dim (int): dimension of the bottleneck layer before the softmax layer for label generation
        emb_dim (int): dimension of the embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of the embedding and output layers
        attn_dim (int):
        attn_sharpening_factor (float):
        attn_sigmoid_smoothing (bool):
        attn_conv_out_channels (int):
        attn_conv_kernel_size (int):
        attn_n_heads (int): number of attention heads
        dropout (float): dropout probability for the RNN layer
        dropout_emb (float): dropout probability for the embedding layer
        dropout_att (float): dropout probability for attention distributions
        lsm_prob (float): label smoothing probability
        ss_prob (float): scheduled sampling probability
        ss_type (str): constant/saturation
        focal_loss_weight (float):
        focal_loss_gamma (float):
        ctc_weight (float):
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        backward (bool): decode in the backward order
        lm_fusion (RNNLM):
        lm_fusion_type (str): type of LM fusion
        discourse_aware (str): state_carry_over/hierarchical
        lm_init (RNNLM):
        global_weight (float):
        mtl_per_batch (bool):
        adaptive_softmax (bool):
        param_init (float):
        mocha_chunk_size (int): chunk size for MoChA
        replace_sos (bool):

    """

    def __init__(self,
                 eos,
                 unk,
                 pad,
                 blank,
                 enc_n_units,
                 attn_type,
                 rnn_type,
                 n_units,
                 n_projs,
                 n_layers,
                 residual,
                 bottleneck_dim,
                 emb_dim,
                 vocab,
                 tie_embedding=False,
                 attn_dim=0,
                 attn_sharpening_factor=0.0,
                 attn_sigmoid_smoothing=False,
                 attn_conv_out_channels=0,
                 attn_conv_kernel_size=0,
                 attn_n_heads=0,
                 dropout=0.0,
                 dropout_emb=0.0,
                 dropout_att=0.0,
                 lsm_prob=0.0,
                 ss_prob=0.0,
                 ss_type='constant',
                 focal_loss_weight=0.0,
                 focal_loss_gamma=2.0,
                 ctc_weight=0.0,
                 ctc_lsm_prob=0.0,
                 ctc_fc_list=[],
                 backward=False,
                 lm_fusion=None,
                 lm_fusion_type='cold',
                 discourse_aware='',
                 lm_init=None,
                 global_weight=1.0,
                 mtl_per_batch=False,
                 adaptive_softmax=False,
                 param_init=0.1,
                 mocha_chunk_size=1,
                 replace_sos=False,
                 soft_label_weight=0.0):

        super(RNNDecoder, self).__init__()
        logger = logging.getLogger('training')

        self.eos = eos
        self.unk = unk
        self.pad = pad
        self.blank = blank
        self.vocab = vocab
        self.attn_type = attn_type
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm', 'gru']
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.residual = residual
        self.ss_prob = ss_prob
        self.ss_type = ss_type
        if ss_type == 'constant':
            self._ss_prob = ss_prob
        elif ss_type == 'saturation':
            self._ss_prob = 0  # start from 0
        self.lsm_prob = lsm_prob
        self.focal_loss_weight = focal_loss_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.ctc_weight = ctc_weight
        self.bwd = backward
        self.lm_fusion_type = lm_fusion_type
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch
        self.replace_sos = replace_sos
        self.soft_label_weight = soft_label_weight

        # for contextualization
        self.discourse_aware = discourse_aware
        self.dstate_prev = None

        # for cache
        self.fifo_cache_ids = []
        self.fifo_cache_sp_key = None
        self.fifo_cache_lm_key = None
        self.dict_cache_sp = {}
        self.static_cache = {}
        self.static_cache_utt_ids = []
        self.dict_cache_lm = {}
        self.prev_spk = ''
        self.total_step = 0
        self.dstates_final = None
        self.lmstate_final = None

        if ctc_weight > 0:
            self.ctc = CTC(eos=eos,
                           blank=blank,
                           enc_n_units=enc_n_units,
                           vocab=vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=param_init)

        if ctc_weight < global_weight:
            # Attention layer
            if attn_type == 'mocha':
                assert attn_n_heads == 1
                self.score = MoChA(key_dim=self.enc_n_units,
                                   query_dim=n_units if n_projs == 0 else n_projs,
                                   attn_dim=attn_dim,
                                   window=mocha_chunk_size,
                                   init_r=-4)
            else:
                if attn_n_heads > 1:
                    self.score = MultiheadAttentionMechanism(
                        key_dim=self.enc_n_units,
                        query_dim=n_units if n_projs == 0 else n_projs,
                        attn_type=attn_type,
                        attn_dim=attn_dim,
                        n_heads=attn_n_heads,
                        dropout=dropout_att)
                else:
                    self.score = AttentionMechanism(
                        key_dim=self.enc_n_units,
                        query_dim=n_units if n_projs == 0 else n_projs,
                        attn_type=attn_type,
                        attn_dim=attn_dim,
                        sharpening_factor=attn_sharpening_factor,
                        sigmoid_smoothing=attn_sigmoid_smoothing,
                        conv_out_channels=attn_conv_out_channels,
                        conv_kernel_size=attn_conv_kernel_size,
                        dropout=dropout_att)

            # Decoder
            self.rnn = nn.ModuleList()
            if self.n_projs > 0:
                self.proj = nn.ModuleList([Linear(n_units, n_projs) for _ in range(n_layers)])
            self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(n_layers)])
            cell = nn.LSTMCell if rnn_type == 'lstm' else nn.GRUCell
            # 1st layer
            dec_idim = enc_n_units + emb_dim
            self.rnn += [cell(dec_idim, n_units)]
            dec_idim = n_units
            if self.n_projs > 0:
                dec_idim = n_projs
            # 2nd layer
            if n_layers >= 2:
                self.rnn += [cell(dec_idim, n_units)]
                if self.n_projs > 0:
                    dec_idim = n_projs
            # 3rd~ layers
            if n_layers >= 3:
                for l in range(n_layers - 2):
                    self.rnn += [cell(dec_idim, n_units)]
                    if self.n_projs > 0:
                        dec_idim = n_projs

            # LM fusion
            if lm_fusion is not None:
                self.linear_dec_feat = Linear(dec_idim + enc_n_units, n_units)
                if lm_fusion_type in ['cold', 'deep']:
                    self.linear_lm_feat = Linear(lm_fusion.n_units, n_units)
                    self.linear_lm_gate = Linear(n_units * 2, n_units)
                elif lm_fusion_type == 'cold_prob':
                    self.linear_lm_feat = Linear(lm_fusion.vocab, n_units)
                    self.linear_lm_gate = Linear(n_units * 2, n_units)
                else:
                    raise ValueError(lm_fusion_type)
                self.output_bn = Linear(n_units * 2, bottleneck_dim)

                # fix LM parameters
                for p in lm_fusion.parameters():
                    p.requires_grad = False
            elif discourse_aware == 'hierarchical':
                raise NotImplementedError
            else:
                self.output_bn = Linear(dec_idim + enc_n_units, bottleneck_dim)

            self.embed = Embedding(vocab, emb_dim,
                                   dropout=dropout_emb,
                                   ignore_index=pad)

            if adaptive_softmax:
                assert self.vocab >= 25000
                self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                    bottleneck_dim, vocab,
                    cutoffs=[self.vocab // 10, 3 * self.vocab // 10],
                    # cutoffs=[self.vocab // 25, 3 * self.vocab // 5],
                    div_value=4.0)
                self.output = None
            else:
                self.adaptive_softmax = None
                self.output = Linear(bottleneck_dim, vocab)
                # NOTE: include bias even when tying weights

                # Optionally tie weights as in:
                # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                # https://arxiv.org/abs/1608.05859
                # and
                # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
                # https://arxiv.org/abs/1611.01462
                if tie_embedding:
                    if emb_dim != bottleneck_dim:
                        raise ValueError('When using the tied flag, n_units must be equal to emb_dim.')
                    self.output.fc.weight = self.embed.embed.weight

        # Initialize parameters
        self.reset_parameters(param_init)

        # resister the external LM
        self.lm = lm_fusion

        # decoder initialization with pre-trained LM
        if lm_init is not None:
            assert lm_init.vocab == vocab
            assert lm_init.n_units == n_units
            assert lm_init.emb_dim == emb_dim
            logger.info('===== Initialize the decoder with pre-trained RNNLM')
            assert lm_init.n_projs == 0  # TODO(hirofumi): fix later
            assert lm_init.n_units_null_context == enc_n_units

            # RNN
            if lm_init.fast_impl:
                for n, p in lm_init.rnn.named_parameters():
                    l = int(n.split('_')[2].replace('l', ''))
                    n = '_'.join(n.split('_')[:2])
                    assert getattr(self.rnn[l], n).size() == p.size()
                    getattr(self.rnn[l], n).data = p.data
                    logger.info('Overwrite %s' % n)
            else:
                for l in range(lm_init.n_layers):
                    for n, p in lm_init.rnn[l].named_parameters():
                        assert getattr(self.rnn[l], n).size() == p.size()
                        getattr(self.rnn[l], n).data = p.data
                        logger.info('Overwrite %s' % n)

            # embedding
            assert self.embed.embed.weight.size() == lm_init.embed.embed.weight.size()
            self.embed.embed.weight.data = lm_init.embed.embed.weight.data
            logger.info('Overwrite %s' % 'embed.embed.weight')

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():

            if 'score.monotonic_energy.v.weight_g' in n or 'score.monotonic_energy.r' in n:
                continue
            if 'score.chunk_energy.v.weight_g' in n or 'score.chunk_energy.r' in n:
                continue

            if p.dim() == 1:
                if 'linear_lm_gate.fc.bias' in n:
                    # Initialize bias in gating with -1 for cold fusion
                    nn.init.constant_(p, val=-1)  # bias
                    logger.info('Initialize %s with %s / %.3f' % (n, 'constant', -1))
                else:
                    nn.init.constant_(p, val=0)  # bias
                    logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0))
            elif p.dim() in [2, 3, 4]:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError

    def start_scheduled_sampling(self):
        self._ss_prob = self.ss_prob

    def forward(self, eouts, elens, ys, task='all', ys_hist=[], teacher_logits=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all/ys/ys_sub*
            ys_hist (list):
            teacher_logits (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None,
                       'acc_att': None, 'ppl_att': None}
        w = next(self.parameters())
        loss = w.new_zeros((1,))

        # if self.lm is not None:
        #     self.lm.eval()

        # CTC loss
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc = self.ctc(eouts, elens, ys)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or ('ctc' not in task)):
            GPUtil.showUtilization()
            loss_att, acc_att, ppl_att = self.forward_att(eouts, elens, ys, ys_hist,
                                                          teacher_logits=teacher_logits)
            observation['loss_att'] = loss_att.item()
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * (self.global_weight - self.ctc_weight)

        observation['loss'] = loss.item()
        return loss, observation

    def append_sos_eos(self, ys, bwd=False, replace_sos=False):
        """Append <sos> and <eos> and return padded sequences.
        Args:
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            ys_in_pad (LongTensor): `[B, L]`
            ys_out_pad (LongTensor): `[B, L]`
            ylens (IntTensor): `[B]`

        """
        w = next(self.parameters())
        eos = w.new_zeros(1).fill_(self.eos).long()
        ys = [np2tensor(np.fromiter(y[::-1] if bwd else y, dtype=np.int64),
                        self.device_id) for y in ys]
        if replace_sos:
            ylens = np2tensor(np.fromiter([y[1:].size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
            ys_in_pad = pad_list([y for y in ys], self.pad)
            ys_out_pad = pad_list([torch.cat([y[1:], eos], dim=0) for y in ys], self.pad)
        else:
            ylens = np2tensor(np.fromiter([y.size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
            ys_in_pad = pad_list([torch.cat([eos, y], dim=0) for y in ys], self.pad)
            ys_out_pad = pad_list([torch.cat([y, eos], dim=0) for y in ys], self.pad)
        return ys_in_pad, ys_out_pad, ylens

    def forward_att(self, eouts, elens, ys, ys_hist=[],
                    return_logits=False, teacher_logits=None):
        """Compute XE loss for the attention-based sequence-to-sequence model.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            ys_hist (list):
            return_logits (bool): return logits for knowledge distillation
            teacher_logits (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity

        """
        bs, xmax = eouts.size()[:2]

        # Append <sos> and <eos>
        ys_in_pad, ys_out_pad, ylens = self.append_sos_eos(ys, self.bwd)

        # Initialization
        if self.discourse_aware == 'hierarchical':
            raise NotImplementedError
        else:
            dstates = self.zero_state(bs)
        if self.discourse_aware == 'state_carry_over' and self.dstate_prev is not None:
            dstates['dstate']['hxs'], dstates['dstate']['cxs'] = self.dstate_prev
            self.dstate_prev = None
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        self.score.reset()
        aw, aws = None, []
        lmout, lmstate = None, None
        GPUtil.showUtilization()
        # Pre-computation of embedding
        ys_emb = self.embed(ys_in_pad)

        # Create the attention mask
        mask = make_pad_mask(elens, self.device_id).expand(bs, xmax)
        GPUtil.showUtilization()
        logits = []
        for t in range(ys_in_pad.size(1)):
            is_sample = t > 0 and self._ss_prob > 0 and random.random() < self._ss_prob and self.adaptive_softmax is None

            # Update LM states for LM fusion
            if self.lm is not None:
                y_lm = self.output(logits[-1]).detach().argmax(-1) if is_sample else ys_in_pad[:, t:t + 1]
                lmout, lmstate = self.lm.decode(y_lm, lmstate)

            # Recurrency -> Score -> Generate
            y_emb = self.embed(self.output(logits[-1]).detach().argmax(-1)) if is_sample else ys_emb[:, t:t + 1]
            dstates, cv, aw, attn_v, lmfeat = self.decode_step(
                eouts, dstates, cv, y_emb, mask, aw, lmout, mode='parallel')
            if not self.training:
                aws.append(aw.transpose(2, 1).unsqueeze(2))  # `[B, n_heads, 1, T]`
            logits.append(attn_v)

            if self.discourse_aware == 'state_carry_over':
                if self.dstate_prev is None:
                    self.dstate_prev = ([None] * bs, [None] * bs)
                if t in ylens.tolist():
                    for b in ylens.tolist().index(t):
                        self.dstate_prev[0][b] = dstates['dstate']['hxs'][b:b + 1].detach()
                        if self.dec_type == 'lstm':
                            self.dstate_prev[1][b] = dstates['dstate']['cxs'][b:b + 1].detach()
        del ys_emb
        del mask
        del ys_in_pad


        if self.discourse_aware == 'state_carry_over':
            self.dstate_prev[0] = torch.cat(self.dstate_prev[0], dim=1)
            if self.dec_type == 'lstm':
                self.dstate_prev[1] = torch.cat(self.dstate_prev[1], dim=1)

        logits = torch.cat(logits, dim=1)
        if self.adaptive_softmax is None:
            logits = self.output(logits)

        # for knowledge distillation
        if return_logits:
            return logits

        # for attention plot
        if not self.training:
            self.aws = tensor2np(torch.cat(aws, dim=2))  # `[B, n_heads, L, T]`
        for t in aws:
            del t 
        torch.cuda.empty_cache()
        # Compute XE sequence loss
        if self.adaptive_softmax is None:
            if self.lsm_prob > 0 and self.training:
                # Label smoothing
                GPUtil.showUtilization()
                loss = cross_entropy_lsm(logits.view((-1, logits.size(2))), ys_out_pad.view(-1),
                                         self.lsm_prob, self.pad)
            else:
                loss = F.cross_entropy(logits.view((-1, logits.size(2))), ys_out_pad.view(-1),
                                       ignore_index=self.pad, size_average=True)

            # Focal loss
            if self.focal_loss_weight > 0:
                fl = focal_loss(logits, ys_out_pad, ylens,
                                alpha=self.focal_loss_weight,
                                gamma=self.focal_loss_gamma)
                loss = loss * (1 - self.focal_loss_weight) + fl * self.focal_loss_weight

            # Knowledge distillation
            if teacher_logits is not None:
                kl_loss = distillation(logits, teacher_logits, ylens, temperature=5.0)
                loss = loss * (1 - self.soft_label_weight) + kl_loss * self.soft_label_weight
        else:
            loss = self.adaptive_softmax(logits.view((-1, logits.size(2))),
                                         ys_out_pad.view(-1)).loss

        # Compute token-level accuracy in teacher-forcing
        if self.adaptive_softmax is None:
            acc = compute_accuracy(logits, ys_out_pad, self.pad)
        else:
            acc = compute_accuracy(self.adaptive_softmax.log_prob(
                logits.view((-1, logits.size(2)))), ys_out_pad, self.pad)
        ppl = np.exp(loss.item())
        # scale loss for CTC
        loss *= ylens.float().mean()

        return loss, acc, ppl

    def decode_step(self, eouts, dstates, cv, y_emb, mask, aw, lmout, mode='hard'):
        dstates = self.recurrency(y_emb, cv, dstates['dstate'])
        cv, aw = self.score(eouts, eouts, dstates['dout_score'], mask, aw, mode)
        attn_v, lmfeat = self.generate(cv, dstates['dout_gen'], lmout)
        return dstates, cv, aw, attn_v, lmfeat

    def zero_state(self, batch_size):
        """Initialize decoder state.

        Args:
            batch_size (int): batch size
        Returns:
            dstates (dict):
                dout (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):

        """
        dstates = {'dout_score': None,  # for attention scoring (1st layer)
                   'dout_gen': None,  # for token generation (last layer)
                   'dstate': None}
        w = next(self.parameters())
        dstates['dout_score'] = w.new_zeros((batch_size, 1, self.dec_n_units))
        dstates['dout_gen'] = w.new_zeros((batch_size, 1, self.dec_n_units))
        state = w.new_zeros((batch_size, self.dec_n_units))
        hxs = [state for _ in range(self.n_layers)]
        cxs = [state for _ in range(self.n_layers)] if self.rnn_type == 'lstm' else []
        dstates['dstate'] = (hxs, cxs)
        return dstates

    def recurrency(self, y_emb, cv, dstate):
        """Recurrency function.

        Args:
            y_emb (FloatTensor): `[B, 1, emb_dim]`
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dstate (tuple): A tuple of (hxs, cxs)
        Returns:
            dstates_new (dict):
                dout_score (FloatTensor): `[B, 1, n_units]`
                dout_gen (FloatTensor): `[B, 1, n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):

        """
        hxs, cxs = dstate
        y_emb = y_emb.squeeze(1)
        cv = cv.squeeze(1)

        dstates_new = {'dout_score': None,  # for attention scoring
                       'dout_gen': None,  # for token generation
                       'dstate': None}

        # 1st layer
        if self.rnn_type == 'lstm':
            hxs[0], cxs[0] = self.rnn[0](torch.cat([y_emb, cv], dim=-1), (hxs[0], cxs[0]))
        elif self.rnn_type == 'gru':
            hxs[0] = self.rnn[0](torch.cat([y_emb, cv], dim=-1), hxs[0])
        dout = self.dropout[0](hxs[0])
        if self.n_projs > 0:
            dout = torch.tanh(self.proj[0](dout))

        # use output in the first layer for attention scoring
        dstates_new['dout_score'] = dout.unsqueeze(1)

        # after 2nd layers
        for l in range(1, self.n_layers):
            if self.rnn_type == 'lstm':
                hxs[l], cxs[l] = self.rnn[l](dout, (hxs[l], cxs[l]))
            elif self.rnn_type == 'gru':
                hxs[l] = self.rnn[l](dout, hxs[l])
            dout_tmp = self.dropout[l](hxs[l])
            if self.n_projs > 0:
                dout_tmp = torch.tanh(self.proj[l](dout_tmp))
            if self.residual:
                dout = dout_tmp + dout
            else:
                dout = dout_tmp

        # use oupput in the the last layer for label generation
        dstates_new['dout_gen'] = dout.unsqueeze(1)
        dstates_new['dstate'] = (hxs[:], cxs[:])
        return dstates_new

    def generate(self, cv, dout, lmout):
        """Generate function.

        Args:
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dout (FloatTensor): `[B, 1, dec_n_units]`
            lmout (FloatTensor): `[B, 1, lm_n_units]`
        Returns:
            attn_v (FloatTensor): `[B, 1, vocab]`
            gated_lmfeat (FloatTensor): `[B, 1 , dec_n_units]`

        """
        gated_lmfeat = None
        if self.lm is not None:
            # LM fusion
            dec_feat = self.linear_dec_feat(torch.cat([dout, cv], dim=-1))

            if self.lm_fusion_type in ['cold', 'deep']:
                lmfeat = self.linear_lm_feat(lmout)
                gate = torch.sigmoid(self.linear_lm_gate(torch.cat([dec_feat, lmfeat], dim=-1)))
                gated_lmfeat = gate * lmfeat
            elif self.lm_fusion_type == 'cold_prob':
                lmfeat = self.linear_lm_feat(self.lm.output(lmout))
                gate = torch.sigmoid(self.linear_lm_gate(torch.cat([dec_feat, lmfeat], dim=-1)))
                gated_lmfeat = gate * lmfeat

            out = self.output_bn(torch.cat([dec_feat, gated_lmfeat], dim=-1))
        else:
            out = self.output_bn(torch.cat([dout, cv], dim=-1))
        attn_v = torch.tanh(out)
        return attn_v, gated_lmfeat

    def _plot_attention(self, save_path, n_cols=1):
        """Plot attention for each head."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        _save_path = mkdir_join(save_path, 'dec_att_weights')

        # Clean directory
        if _save_path is not None and os.path.isdir(_save_path):
            shutil.rmtree(_save_path)
            os.mkdir(_save_path)

        if hasattr(self, 'aws'):
            plt.clf()
            fig, axes = plt.subplots(max(1, self.score.n_heads // n_cols), n_cols,
                                     figsize=(20, 8), squeeze=False)
            for h in range(self.score.n_heads):
                ax = axes[h // n_cols, h % n_cols]
                ax.imshow(self.aws[-1,  h, :, :], aspect="auto")
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            fig.tight_layout()
            fig.savefig(os.path.join(_save_path, 'attention.png'), dvi=500)
            plt.close()

    def greedy(self, eouts, elens, max_len_ratio, idx2token,
               exclude_eos=False, oracle=False,
               refs_id=None, utt_ids=None, speakers=None):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            oracle (bool): teacher-forcing mode
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
        Returns:
            hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aws (list): A list of length `[B]`, which contains arrays of size `[n_heads, L, T]`

        """
        logger = logging.getLogger("decoding")
        bs, xmax, _ = eouts.size()

        # Initialization
        dstates = self.zero_state(bs)
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        self.score.reset()
        aw = None
        lmout, lmstate = None, None
        y = eouts.new_zeros(bs, 1).fill_(refs_id[0][0] if self.replace_sos else self.eos)

        # Create the attention mask
        mask = make_pad_mask(elens, self.device_id).expand(bs, xmax)

        hyps_batch, aws_batch = [], []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        if oracle:
            assert refs_id is not None
            ymax = max([len(refs_id[b]) for b in range(bs)]) + 1
        else:
            ymax = int(math.floor(xmax * max_len_ratio)) + 1
        for t in range(ymax):
            # Update LM states for LM fusion
            if self.lm is not None:
                lmout, lmstate = self.lm.decode(self.lm(y), lmstate)

            # Recurrency -> Score -> Generate
            y_emb = self.embed(y)
            dstates, cv, aw, attn_v, lmfeat = self.decode_step(
                eouts, dstates, cv, y_emb, mask, aw, lmout)
            aws_batch += [aw.transpose(2, 1).unsqueeze(2)]  # `[B, n_heads, 1, T]`

            # Pick up 1-best
            if self.adaptive_softmax is None:
                y = self.output(attn_v).argmax(-1)
            else:
                y = self.adaptive_softmax.predict(attn_v.view(-1, attn_v.size(2))).unsqueeze(1)
            hyps_batch += [y]

            # Count lengths of hypotheses
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                    ylens[b] += 1  # include <eos>

            # Break if <eos> is outputed in all mini-batch
            if sum(eos_flags) == bs:
                break
            if t == ymax - 1:
                break

            if oracle:
                y = eouts.new_zeros(bs, 1)
                for b in range(bs):
                    y[b] = refs_id[b][t]

        # LM state carry over
        self.lmstate_final = lmstate

        # Concatenate in L dimension
        hyps_batch = tensor2np(torch.cat(hyps_batch, dim=1))
        aws_batch = tensor2np(torch.cat(aws_batch, dim=2))  # `[B, n_heads, L, T]`

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.bwd:
            # Reverse the order
            hyps = [hyps_batch[b, :ylens[b]][::-1] for b in range(bs)]
            aws = [aws_batch[b, :, :ylens[b]][::-1] for b in range(bs)]
        else:
            hyps = [hyps_batch[b, :ylens[b]] for b in range(bs)]
            aws = [aws_batch[b, :, :ylens[b]] for b in range(bs)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.bwd:
                hyps = [hyps[b][1:] if eos_flags[b] else hyps[b] for b in range(bs)]
            else:
                hyps = [hyps[b][:-1] if eos_flags[b] else hyps[b] for b in range(bs)]

        for b in range(bs):
            if utt_ids is not None:
                logger.info('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.info('Ref: %s' % idx2token(refs_id[b]))
            if self.bwd:
                logger.info('Hyp: %s' % idx2token(hyps[b][1:][::-1]))
            else:
                logger.info('Hyp: %s' % idx2token(hyps[b][1:]))

        return hyps, aws

    def beam_search(self, eouts, elens, params, idx2token,
                    lm=None, lm_rev=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False,
                    refs_id=None, utt_ids=None, speakers=None,
                    ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[]):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            params (dict):
                recog_beam_width (int): size of beam
                recog_max_len_ratio (int): maximum sequence length of tokens
                recog_min_len_ratio (float): minimum sequence length of tokens
                recog_length_penalty (float): length penalty
                recog_coverage_penalty (float): coverage penalty
                recog_coverage_threshold (float): threshold for coverage penalty
                recog_lm_weight (float): weight of LM score
                recog_n_caches (int):
            idx2token (): converter from index to token
            lm (RNNLM/GatedConvLM/TransformerLM):
            lm_rev (RNNLM/GatedConvLM/TransformerLM):
            ctc_log_probs (FloatTensor):
            nbest (int):
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
            ensmbl_eouts (list): list of FloatTensor
            ensmbl_elens (list) list of list
            ensmbl_decs (list): list of torch.nn.Module
        Returns:
            nbest_hyps_idx (list): A list of length `[B]`, which contains list of N hypotheses
            aws (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            scores (list):
            cache_info (tuple):

        """
        logger = logging.getLogger("decoding")

        bs = eouts.size(0)
        n_models = len(ensmbl_decs) + 1

        oracle = params['recog_oracle']
        beam_width = params['recog_beam_width']
        ctc_weight = params['recog_ctc_weight']
        max_len_ratio = params['recog_max_len_ratio']
        min_len_ratio = params['recog_min_len_ratio']
        lp_weight = params['recog_length_penalty']
        cp_weight = params['recog_coverage_penalty']
        cp_threshold = params['recog_coverage_threshold']
        lm_weight = params['recog_lm_weight']
        gnmt_decoding = params['recog_gnmt_decoding']
        eos_threshold = params['recog_eos_threshold']
        asr_state_carry_over = params['recog_asr_state_carry_over']
        lm_state_carry_over = params['recog_lm_state_carry_over']
        n_caches = params['recog_n_caches']
        cache_theta_sp = params['recog_cache_theta_speech']
        cache_lambda_sp = params['recog_cache_lambda_speech']
        cache_theta_lm = params['recog_cache_theta_lm']
        cache_lambda_lm = params['recog_cache_lambda_lm']
        cache_type = params['recog_cache_type']

        if lm is not None:
            lm.eval()
        if lm_rev is not None:
            lm_rev.eval()

        # For joint CTC-Attention decoding
        if ctc_weight > 0 and ctc_log_probs is not None:
            if self.bwd:
                ctc_prefix_score = CTCPrefixScore(tensor2np(ctc_log_probs)[0][::-1], self.blank, self.eos)
            else:
                ctc_prefix_score = CTCPrefixScore(tensor2np(ctc_log_probs)[0], self.blank, self.eos)

        nbest_hyps_idx, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            # Initialization per utterance
            dstates = self.zero_state(1)
            cv = eouts.new_zeros(1, 1, self.enc_n_units)
            self.score.reset()
            lmstate = None

            # Ensemble initialization
            ensmbl_dstate, ensmbl_cv = [], []
            if n_models > 1:
                for dec in ensmbl_decs:
                    ensmbl_dstate += [dec.zero_state(1)]
                    ensmbl_cv += [eouts.new_zeros(1, 1, dec.enc_n_units)]
                    dec.score.reset()

            if speakers is not None and speakers[b] != self.prev_spk:
                self.reset_global_cache()
            else:
                if lm_state_carry_over and isinstance(lm, RNNLM):
                    lmstate = self.lmstate_final
                if asr_state_carry_over:
                    dstates = self.dstates_final
            self.prev_spk = speakers[b]

            end_hyps = []
            hyps = [{'hyp': [self.eos],
                     'score': 0.0,
                     'hist_score': [0.0],
                     'score_attn': 0.0,
                     'score_ctc': 0.0,
                     'score_lm': 0.0,
                     'dstates': dstates,
                     'cv': cv,
                     'aws': [None],
                     'lmstate': lmstate,
                     'ensmbl_dstate': ensmbl_dstate,
                     'ensmbl_cv': ensmbl_cv,
                     'ensmbl_aws':[[None]] * (n_models - 1),
                     'ctc_state': ctc_prefix_score.initial_state() if ctc_weight > 0 and ctc_log_probs is not None else None,
                     'ctc_score': 0.0,
                     'cache_ids': [],
                     'dict_cache_sp': {},
                     'cache_sp_key': [],
                     'cache_lm_key': [],
                     'cache_idx_hist': [],
                     'cache_sp_attn_hist': [],
                     'cache_lm_attn_hist': [],
                     }]
            if oracle:
                assert refs_id is not None
                ymax = len(refs_id[b]) + 1
            else:
                ymax = int(math.floor(elens[b] * max_len_ratio)) + 1
            for t in range(ymax):
                new_hyps = []
                for beam in hyps:
                    if self.replace_sos and t == 0:
                        prev_idx = refs_id[0][0]
                    else:
                        prev_idx = ([self.eos] + refs_id[b])[t] if oracle else beam['hyp'][-1]

                    if self.lm is not None:
                        # Update LM states for LM fusion
                        lmout, lmstate, lm_log_probs = self.lm.predict(
                            eouts.new_zeros(1, 1).fill_(prev_idx), beam['lmstate'])
                    elif lm_weight > 0 and lm is not None:
                        # Update LM states for shallow fusion
                        lmout, lmstate, lm_log_probs = lm.predict(
                            eouts.new_zeros(1, 1).fill_(prev_idx), beam['lmstate'])
                    else:
                        lmout, lmstate, lm_log_probs = None, None, None

                    # for the main model
                    y_emb = self.embed(eouts.new_zeros(1, 1).fill_(prev_idx))
                    dstates, cv, aw, attn_v, lmfeat = self.decode_step(
                        eouts[b:b + 1, :elens[b]],
                        beam['dstates'], beam['cv'], y_emb,
                        None,  # mask
                        beam['aws'][-1], lmout)
                    if self.adaptive_softmax is None:
                        probs = F.softmax(self.output(attn_v).squeeze(1), dim=1)
                    else:
                        probs = self.adaptive_softmax.log_prob(attn_v.view(-1, attn_v.size(2)))

                    # for the ensemble
                    ensmbl_dstate, ensmbl_cv, ensmbl_aws = [], [], []
                    ensmbl_log_probs = 0.
                    if n_models > 1:
                        for i_e, dec in enumerate(ensmbl_decs):
                            y_emb = dec.embed(eouts.new_zeros(1, 1).fill_(prev_idx))
                            ensmbl_dstate, cv_e, aw_e, attn_v_e, _ = dec.decode_step(
                                ensmbl_eouts[i_e][b:b + 1, :ensmbl_elens[i_e][b]],
                                beam['ensmbl_dstate'][i_e], beam['ensmbl_cv'], y_emb,
                                None,  # mask
                                beam['ensmbl_aws'][-1], lmout)

                            ensmbl_dstate += [ensmbl_dstate]
                            ensmbl_cv += [cv_e]
                            ensmbl_aws += [aw_e.unsqueeze(0)]  # TODO(hirofumi): unsqueeze?
                            if dec.adaptive_softmax is None:
                                ensmbl_log_probs += F.log_softmax(dec.output(attn_v_e).squeeze(1), dim=1)
                            else:
                                ensmbl_log_probs += dec.adaptive_softmax.log_prob(attn_v_e.view(-1, attn_v_e.size(2)))

                    # Cache decoding
                    cache_ids = None
                    cache_sp_attn = None
                    cache_lm_attn = None
                    cache_sp_probs = probs.new_zeros(probs.size())
                    cache_lm_probs = probs.new_zeros(probs.size())
                    if n_caches > 0:
                        assert self.adaptive_softmax is None

                        # Compute inner-product over caches
                        if 'speech_fifo' in cache_type:
                            is_cache = True
                            if 'online' in cache_type and len(self.fifo_cache_ids + beam['cache_ids']) > 0:
                                cache_ids = (self.fifo_cache_ids + beam['cache_ids'])[-n_caches:]
                                cache_sp_key = beam['cache_sp_key'][-n_caches:]
                                if len(cache_sp_key) > 0:
                                    cache_sp_key = torch.cat(cache_sp_key, dim=1)
                                    if self.fifo_cache_sp_key is not None:
                                        cache_sp_key = torch.cat([self.fifo_cache_sp_key, cache_sp_key], dim=1)
                                else:
                                    cache_sp_key = self.fifo_cache_sp_key  # for the first token
                                # Truncate
                                cache_sp_key = cache_sp_key[:, -n_caches:]  # `[1, L, enc_n_units]`
                            elif 'online' not in cache_type and len(self.fifo_cache_ids) > 0:
                                cache_ids = self.fifo_cache_ids
                                cache_sp_key = self.fifo_cache_sp_key
                            else:
                                is_cache = False

                            if is_cache:
                                cache_sp_attn = F.softmax(cache_theta_sp * torch.matmul(
                                    cache_sp_key, torch.cat([cv, dstates['dout_gen']], dim=-1).transpose(2, 1)), dim=1)  # `[1, L, 1]`
                                # Sum all probabilities
                                for c in set(beam['cache_ids']):
                                    for offset in [i for i, key in enumerate(cache_ids) if key == c]:
                                        cache_sp_probs[0, c] += cache_sp_attn[0, offset, 0]
                                probs = (1 - cache_lambda_sp) * probs + cache_lambda_sp * cache_sp_probs

                        if 'lm_fifo' in cache_type:
                            is_cache = True
                            if 'online' in cache_type and len(self.fifo_cache_ids + beam['cache_ids']) > 0:
                                assert lm_weight > 0
                                cache_ids = (self.fifo_cache_ids + beam['cache_ids'])[-n_caches:]
                                cache_lm_key = beam['cache_lm_key'][-n_caches:]
                                if len(cache_lm_key) > 0:
                                    cache_lm_key = torch.cat(cache_lm_key, dim=1)
                                    if self.fifo_cache_lm_key is not None:
                                        cache_lm_key = torch.cat([self.fifo_cache_lm_key, cache_lm_key], dim=1)
                                else:
                                    cache_lm_key = self.fifo_cache_lm_key   # for the first token
                                # Truncate
                                cache_lm_key = cache_lm_key[:, -n_caches:]  # `[1, L, lm_n_units]`
                            elif 'online' not in cache_type and len(self.fifo_cache_ids) > 0:
                                cache_ids = self.fifo_cache_ids
                                cache_lm_key = self.fifo_cache_lm_key
                            else:
                                is_cache = False

                            if is_cache:
                                cache_lm_attn = F.softmax(cache_theta_lm * torch.matmul(
                                    cache_lm_key, lmout.transpose(2, 1)), dim=1)  # `[1, L, 1]`
                                # Sum all probabilities
                                for c in set(beam['cache_ids']):
                                    for offset in [i for i, key in enumerate(cache_ids) if key == c]:
                                        cache_lm_probs[0, c] += cache_lm_attn[0, offset, 0]
                                lm_probs = torch.exp(lm_log_probs)[:, -1]
                                lm_probs = (1 - cache_lambda_lm) * lm_probs + cache_lambda_lm * cache_lm_probs
                                lm_log_probs = torch.log(lm_probs).unsqueeze(1)

                        if 'speech_dict' in cache_type and len(self.dict_cache_sp.keys()) > 0:
                            cache_ids = sorted(list(self.dict_cache_sp.keys()))
                            cache_ids = [self.unk if idx < 0 else idx for idx in cache_ids]
                            cache_sp_key = [v['key']for k, v in sorted(self.dict_cache_sp.items(), key=lambda x: x[0])]
                            cache_sp_key = torch.cat(cache_sp_key, dim=1)
                            cache_sp_attn = F.softmax(cache_theta_sp * torch.matmul(
                                cache_sp_key, torch.cat([cv, dstates['dout_gen']], dim=-1).transpose(2, 1)), dim=1)  # `[1, L, 1]`
                            # Sum all probabilities
                            for offset, c in enumerate(cache_ids):
                                cache_sp_probs[0, c] += cache_sp_attn[0, offset, 0]
                            probs = (1 - cache_lambda_sp) * probs + cache_lambda_sp * cache_sp_probs

                        if 'lm_dict' in cache_type and len(self.dict_cache_lm.keys()) > 0:
                            cache_ids = sorted(list(self.dict_cache_lm.keys()))
                            cache_lm_key = [v['key']for k, v in sorted(self.dict_cache_lm.items(), key=lambda x: x[0])]
                            cache_lm_key = torch.cat(cache_lm_key, dim=1)
                            cache_lm_attn = F.softmax(cache_theta_lm * torch.matmul(
                                cache_lm_key, lmout.transpose(2, 1)), dim=1)  # `[1, L, 1]`
                            # Sum all probabilities
                            for offset, c in enumerate(cache_ids):
                                cache_lm_probs[0, c] += cache_lm_attn[0, offset, 0]
                            probs = (1 - cache_lambda_lm) * probs + cache_lambda_lm * cache_lm_probs

                    if self.adaptive_softmax is None:
                        local_scores_attn = torch.log(probs)
                    else:
                        local_scores_attn = probs  # NOTE: already log-scaled

                    # Ensemble in log-scale
                    if n_models > 1:
                        local_scores_attn += ensmbl_log_probs
                        local_scores_attn /= n_models

                    # Attention scores
                    scores_attn = beam['score_attn'] + local_scores_attn
                    total_scores = scores_attn * (1 - ctc_weight)

                    # Add LM score <after> top-K selection
                    total_scores_topk, topk_ids = torch.topk(
                        total_scores, k=beam_width, dim=1, largest=True, sorted=True)
                    if lm_weight > 0 and lm is not None:
                        total_scores_lm = beam['score_lm'] + lm_log_probs[0, -1, topk_ids[0]]
                        total_scores_topk += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = torch.zeros((beam_width), dtype=torch.float32)

                    # Add length penalty
                    lp = 1.0
                    if lp_weight > 0:
                        if gnmt_decoding:
                            lp = math.pow(6 + len(beam['hyp'][1:]), lp_weight) / math.pow(6, lp_weight)
                            total_scores_topk /= lp
                        else:
                            total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight

                    # Add coverage penalty
                    cp = 0.0
                    aw_mat = None
                    if cp_weight > 0:
                        aw_mat = torch.stack(beam['aws'][1:] + [aw], dim=-1)  # `[B, T, L, n_heads]`
                        aw_mat = aw_mat[:, :, :, 0]
                        if gnmt_decoding:
                            aw_mat = torch.log(aw_mat.sum(-1))
                            cp = torch.where(aw_mat < 0, aw_mat, aw_mat.new_zeros(aw_mat.size())).sum()
                            # TODO(hirofumi): mask by elens[b]
                            total_scores_topk += cp * cp_weight
                        else:
                            # Recompute converage penalty in each step
                            if cp_threshold == 0:
                                cp = aw_mat.sum() / self.score.n_heads
                            else:
                                cp = torch.where(aw_mat > cp_threshold, aw_mat,
                                                 aw_mat.new_zeros(aw_mat.size())).sum() / self.score.n_heads
                            total_scores_topk += cp * cp_weight

                    # CTC score
                    if ctc_weight > 0 and ctc_log_probs is not None:
                        ctc_scores, ctc_states = ctc_prefix_score(
                            beam['hyp'], tensor2np(topk_ids[0]), beam['ctc_state'])
                        total_scores_ctc = torch.from_numpy(ctc_scores).cuda(self.device_id)
                        total_scores_topk += total_scores_ctc * ctc_weight
                        # Sort again
                        total_scores_topk, joint_ids_topk = torch.topk(
                            total_scores_topk, k=beam_width, dim=1, largest=True, sorted=True)
                        topk_ids = topk_ids[:, joint_ids_topk[0]]
                    else:
                        total_scores_ctc = torch.zeros((beam_width,), dtype=torch.float32)

                    for k in range(beam_width):
                        idx = topk_ids[0, k].item()
                        total_score = total_scores_topk[0, k].item()

                        # Exclude short hypotheses
                        if idx == self.eos:
                            if len(beam['hyp']) - 1 < elens[b] * min_len_ratio:
                                continue
                            # EOS threshold
                            max_score_no_eos = local_scores_attn[0, :idx].max(0)[0].item()
                            max_score_no_eos = max(max_score_no_eos, local_scores_attn[0, idx + 1:].max(0)[0].item())
                            if local_scores_attn[0, idx].item() <= eos_threshold * max_score_no_eos:
                                continue

                        new_hyps.append(
                            {'hyp': beam['hyp'] + [idx],
                             'score': total_score,
                             'hist_score': beam['hist_score'] + [total_score],
                             'score_attn': scores_attn[0, idx].item(),
                             'score_cp': cp,
                             'score_ctc': total_scores_ctc[k].item(),
                             'score_lm': total_scores_lm[k].item(),
                             'dstates': dstates,
                             'cv': cv,
                             'aws': beam['aws'] + [aw],
                             'lmstate': lmstate,
                             'ensmbl_dstate': ensmbl_dstate,
                             'ensmbl_cv': ensmbl_cv,
                             'ensmbl_aws': ensmbl_aws,
                             'ctc_state': ctc_states[joint_ids_topk[0, k]] if ctc_log_probs is not None else None,
                             'ctc_score': ctc_scores[joint_ids_topk[0, k]] if ctc_log_probs is not None else None,
                             'cache_ids': beam['cache_ids'] + [idx],
                             'cache_sp_key': beam['cache_sp_key'] + [torch.cat([cv, dstates['dout_gen']], dim=-1)],
                             'cache_lm_key': beam['cache_lm_key'] + [lmout],
                             'cache_idx_hist': beam['cache_idx_hist'] + [cache_ids],
                             'cache_sp_attn_hist': beam['cache_sp_attn_hist'] + [cache_sp_attn] if cache_sp_attn is not None else [],
                             'cache_lm_attn_hist': beam['cache_lm_attn_hist'] + [cache_lm_attn] if cache_lm_attn is not None else [],
                             })

                # Local pruning
                new_hyps_tmp = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]

                # Remove complete hypotheses
                new_hyps = []
                for hyp in new_hyps_tmp:
                    if oracle:
                        if t == len(refs_id[b]):
                            end_hyps += [hyp]
                        else:
                            new_hyps += [hyp]
                    else:
                        if hyp['hyp'][-1] == self.eos:
                            end_hyps += [hyp]
                        else:
                            new_hyps += [hyp]
                if len(end_hyps) >= beam_width:
                    end_hyps = end_hyps[:beam_width]
                    break
                hyps = new_hyps[:]

            # Global pruning
            if len(end_hyps) == 0:
                end_hyps = hyps[:]
            elif len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(hyps[:nbest - len(end_hyps)])

            # backward LM rescoring
            if lm_rev is not None and lm_weight > 0:
                for i in range(len(end_hyps)):
                    # Initialize
                    lmstate_rev = None
                    score_lm_rev = 0.0
                    lp = 1.0

                    # Append <eos>
                    if end_hyps[i]['hyp'][-1] != self.eos:
                        end_hyps[i]['hyp'].append(self.eos)
                        logger.info('Append <eos>.')

                    if lp_weight > 0 and gnmt_decoding:
                        lp = math.pow(6 + (len(end_hyps[i]['hyp'][1:])), lp_weight) / math.pow(6, lp_weight)
                    for t in range(len(end_hyps[i]['hyp'][1:])):
                        lmout_rev, lmstate_rev = lm_rev.decode(
                            eouts.new_zeros(1, 1).fill_(end_hyps[i]['hyp'][::-1][t]), lmstate_rev)
                        lm_log_probs = F.log_softmax(lm_rev.generate(lmout_rev).squeeze(1), dim=-1)
                        score_lm_rev += lm_log_probs[0, -1, end_hyps[i]['hyp'][::-1][t + 1]]
                    if gnmt_decoding:
                        score_lm_rev /= lp  # normalize
                    end_hyps[i]['score'] += score_lm_rev * lm_weight
                    end_hyps[i]['score_lm_rev'] = score_lm_rev

            # Sort by score
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)

            # N-best list
            if self.bwd:
                # Reverse the order
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:][::-1]) for n in range(nbest)]]
                aws += [[end_hyps[n]['aws'][1:][::-1] for n in range(nbest)]]
                scores += [[end_hyps[n]['hist_score'][1:][::-1] for n in range(nbest)]]
            else:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
                aws += [[end_hyps[n]['aws'][1:] for n in range(nbest)]]
                scores += [[end_hyps[n]['hist_score'][1:] for n in range(nbest)]]

            # Check <eos>
            eos_flag = [True if end_hyps[n]['hyp'][-1] == self.eos else False for n in range(nbest)]
            eos_flags.append(eos_flag)

            if utt_ids is not None:
                logger.info('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.info('Ref: %s' % idx2token(refs_id[b]))
            for k in range(len(end_hyps)):
                if self.bwd:
                    logger.info('Hyp: %s' % idx2token(end_hyps[k]['hyp'][1:][::-1]))
                else:
                    logger.info('Hyp: %s' % idx2token(end_hyps[k]['hyp'][1:]))
                logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                logger.info('log prob (hyp, att): %.7f' % (end_hyps[k]['score_attn'] * (1 - ctc_weight)))
                logger.info('log prob (hyp, cp): %.7f' % (end_hyps[k]['score_cp'] * cp_weight))
                if ctc_weight > 0 and ctc_log_probs is not None:
                    logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc'] * ctc_weight))
                if lm_weight > 0 and lm is not None:
                    logger.info('log prob (hyp, lm): %.7f' % (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_rev is not None:
                        logger.info('log prob (hyp, lm reverse): %.7f' % (end_hyps[k]['score_lm_rev'] * lm_weight))
                if n_caches > 0:
                    logger.info('Cache: %d' % (len(self.fifo_cache_ids) + len(end_hyps[k]['cache_ids'])))

        # Concatenate in L dimension
        for b in range(len(aws)):
            for n in range(nbest):
                aws[b][n] = tensor2np(torch.stack(aws[b][n], dim=1).squeeze(0))

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.bwd:
                nbest_hyps_idx = [[nbest_hyps_idx[b][n][1:] if eos_flags[b][n]
                                   else nbest_hyps_idx[b][n] for n in range(nbest)] for b in range(bs)]
            else:
                nbest_hyps_idx = [[nbest_hyps_idx[b][n][:-1] if eos_flags[b][n]
                                   else nbest_hyps_idx[b][n] for n in range(nbest)] for b in range(bs)]

        # Store in cache
        cache_idx_hist = None
        cache_sp_attn_hist = None
        cache_lm_attn_hist = None
        if n_caches > 0:
            hyp_len = len(end_hyps[0]['hyp'][1:])

            if 'speech_fifo' in cache_type and len(end_hyps[0]['cache_ids']) > 0:
                self.fifo_cache_ids = (self.fifo_cache_ids + end_hyps[0]['cache_ids'])[-n_caches:]
                cache_idx_hist = end_hyps[0]['cache_idx_hist']
                cache_sp_key = end_hyps[0]['cache_sp_key'][-n_caches:]
                cache_sp_key = torch.cat(cache_sp_key, dim=1)
                if self.fifo_cache_sp_key is not None:
                    cache_sp_key = torch.cat([self.fifo_cache_sp_key, cache_sp_key], dim=1)
                # Truncate
                self.fifo_cache_sp_key = cache_sp_key[:, -n_caches:]  # `[1, L, enc_n_units]`
                if len(end_hyps[0]['cache_sp_attn_hist']) > 0:
                    cache_sp_attn_hist = torch.zeros(
                        (1, end_hyps[0]['cache_sp_attn_hist'][-1].size(1), hyp_len), dtype=torch.float32)
                    for i, p in enumerate(end_hyps[0]['cache_sp_attn_hist']):
                        if p.size(1) < n_caches:
                            cache_sp_attn_hist[0, : p.size(1), i] = p[0, :, 0].cpu()
                        else:
                            cache_sp_attn_hist[0, : n_caches - (hyp_len - 1 - i), i] = p[0, (hyp_len - 1 - i):, 0].cpu()

            if ('lm_fifo' in cache_type) and len(end_hyps[0]['cache_ids']) > 0:
                assert lm_weight > 0
                self.fifo_cache_ids = (self.fifo_cache_ids + end_hyps[0]['cache_ids'])[-n_caches:]
                cache_idx_hist = end_hyps[0]['cache_idx_hist']
                cache_lm_key = end_hyps[0]['cache_lm_key'][-n_caches:]
                cache_lm_key = torch.cat(cache_lm_key, dim=1)
                if self.fifo_cache_lm_key is not None:
                    cache_lm_key = torch.cat([self.fifo_cache_lm_key, cache_lm_key], dim=1)
                # Truncate
                self.fifo_cache_lm_key = cache_lm_key[:, -n_caches:]  # `[1, L, lm_n_units]`
                if len(end_hyps[0]['cache_lm_attn_hist']) > 0:
                    cache_lm_attn_hist = torch.zeros((1, end_hyps[0]['cache_lm_attn_hist'][-1].size(1), hyp_len),
                                                     dtype=torch.float32)  # `[B, n_keys, n_values]`
                    for i, p in enumerate(end_hyps[0]['cache_lm_attn_hist']):
                        if p.size(1) < n_caches:
                            cache_lm_attn_hist[0, : p.size(1), i] = p[0, :, 0].cpu()
                        else:
                            cache_lm_attn_hist[0, : n_caches - (hyp_len - 1 - i), i] = p[0, (hyp_len - 1 - i):, 0].cpu()

            if 'speech_dict' in cache_type:
                if len(self.dict_cache_sp.keys()) > 0:
                    cache_idx_hist = sorted(list(self.dict_cache_sp.keys()))
                    cache_idx_hist = [self.unk if i < 0 else i for i in cache_idx_hist]
                    cache_sp_key = [v['key']for k, v in sorted(self.dict_cache_sp.items(), key=lambda x: x[0])]
                    cache_sp_key = torch.cat(cache_sp_key, dim=1)
                    cache_sp_attn_hist = torch.cat(end_hyps[0]['cache_sp_attn_hist'], dim=-1).cpu().numpy()

                for t, idx in enumerate(end_hyps[0]['hyp'][1:]):
                    if idx == self.eos:
                        continue
                    if idx != self.unk and idx in self.dict_cache_sp.keys():
                        if cache_type in ['speech_dict_overwrite', 'joint_dict_overwrite']:
                            self.dict_cache_sp[idx]['value'] = (
                                self.dict_cache_sp[idx]['value'] + end_hyps[0]['cache_sp_key'][t]) / 2
                        else:
                            self.dict_cache_sp[idx]['value'] = end_hyps[0]['cache_sp_key'][t]
                        self.dict_cache_sp[idx]['time'] = self.total_step + t + 1
                        self.dict_cache_sp[idx]['count'] += 1
                    else:
                        if idx == self.unk:
                            self.dict_cache_sp[idx - (self.total_step + t + 1)] = {
                                'key': end_hyps[0]['cache_sp_key'][t],
                                'value': end_hyps[0]['cache_sp_key'][t],
                                'count': 1,
                                'time': self.total_step + t + 1}
                        else:
                            self.dict_cache_sp[idx] = {
                                'key': end_hyps[0]['cache_sp_key'][t],
                                'value': end_hyps[0]['cache_sp_key'][t],
                                'count': 1,
                                'time': self.total_step + t + 1}
                        if len(self.dict_cache_sp.keys()) > n_caches:
                            oldest_id = sorted(self.dict_cache_sp.items(), key=lambda x: x[1]['time'])[0][0]
                            self.dict_cache_sp.pop(oldest_id)
                self.total_step += len(end_hyps[0]['hyp'][1:])

            if 'lm_dict' in cache_type:
                if len(self.dict_cache_lm.keys()) > 0:
                    cache_idx_hist = sorted(list(self.dict_cache_lm.keys()))
                    cache_lm_key = [v['key']for k, v in sorted(self.dict_cache_lm.items(), key=lambda x: x[0])]
                    cache_lm_key = torch.cat(cache_lm_key, dim=1)
                    cache_lm_attn_hist = torch.cat(end_hyps[0]['cache_lm_attn_hist'], dim=-1).cpu().numpy()

                for t, idx in enumerate(end_hyps[0]['hyp'][1:]):
                    if idx == self.eos:
                        continue
                    if idx in self.dict_cache_lm.keys():
                        if cache_type in ['lm_dict_overwrite', 'joint_dict_overwrite']:
                            self.dict_cache_lm[idx]['value'] = (
                                self.dict_cache_lm[idx]['value'] + end_hyps[0]['cache_lm_key'][t]) / 2
                        else:
                            self.dict_cache_lm[idx]['value'] = end_hyps[0]['cache_lm_key'][t]
                        self.dict_cache_lm[idx]['time'] = self.total_step + t + 1
                        self.dict_cache_lm[idx]['count'] += 1
                    else:
                        self.dict_cache_lm[idx] = {
                            'key': end_hyps[0]['cache_lm_key'][t],
                            'value': end_hyps[0]['cache_lm_key'][t],
                            'count': 1,
                            'time': self.total_step + t + 1}
                        if len(self.dict_cache_lm.keys()) > n_caches:
                            oldest_id = sorted(self.dict_cache_lm.items(), key=lambda x: x[1]['time'])[0][0]
                            self.dict_cache_lm.pop(oldest_id)
                self.total_step += len(end_hyps[0]['hyp'][1:])

        # Store ASR/LM state
        self.dstates_final = end_hyps[0]['dstates']
        self.lmstate_final = end_hyps[0]['lmstate']

        if 'speech' in cache_type:
            return nbest_hyps_idx, aws, scores, (cache_sp_attn_hist, cache_idx_hist)
        else:
            return nbest_hyps_idx, aws, scores, (cache_lm_attn_hist, cache_idx_hist)

    def reset_global_cache(self):
        """Reset global cache when the speaker/session is changed."""
        self.fifo_cache_ids = []
        self.fifo_cache_sp_key = None
        self.fifo_cache_lm_key = None
        self.dict_cache_sp = {}
        self.dict_cache_lm = {}
        self.total_step = 0

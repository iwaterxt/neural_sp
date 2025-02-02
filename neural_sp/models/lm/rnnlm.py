#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Recurrent neural network language model (RNNLM)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.lm.lm_base import LMBase
from neural_sp.models.modules.embedding import Embedding
from neural_sp.models.modules.linear import Linear


class RNNLM(LMBase):
    """RNN language model."""

    def __init__(self, args, save_path=None):

        super(LMBase, self).__init__()
        logger = logging.getLogger('training')
        logger.info(self.__class__.__name__)

        self.save_path = save_path

        self.emb_dim = args.emb_dim
        self.rnn_type = args.lm_type
        assert args.lm_type in ['lstm', 'gru']
        self.n_units = args.n_units
        self.n_projs = args.n_projs
        self.n_layers = args.n_layers
        self.residual = args.residual
        self.use_glu = args.use_glu
        self.n_units_cv = args.n_units_null_context
        self.lsm_prob = args.lsm_prob

        self.vocab = args.vocab
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for cache
        self.cache_theta = 0.2  # smoothing parameter
        self.cache_lambda = 0.2  # cache weight
        self.cache_ids = []
        self.cache_keys = []
        self.cache_attn = []

        self.embed = Embedding(vocab=self.vocab,
                               emb_dim=args.emb_dim,
                               dropout=args.dropout_in,
                               ignore_index=self.pad)

        self.fast_impl = False
        rnn = nn.LSTM if args.lm_type == 'lstm' else nn.GRU
        if args.n_projs == 0 and not args.residual:
            self.fast_impl = True
            self.rnn = rnn(args.emb_dim + args.n_units_null_context,
                           args.n_units, args.n_layers,
                           bias=True,
                           batch_first=True,
                           dropout=args.dropout_hidden,
                           bidirectional=False)
            # NOTE: pytorch introduces a dropout layer on the outputs of each layer EXCEPT the last layer
            rnn_idim = args.n_units
            self.dropout_top = nn.Dropout(p=args.dropout_hidden)
        else:
            self.rnn = nn.ModuleList()
            self.dropout = nn.ModuleList([nn.Dropout(p=args.dropout_hidden)
                                          for _ in range(args.n_layers)])
            if args.n_projs > 0:
                self.proj = nn.ModuleList([Linear(args.n_units, args.n_projs)
                                           for _ in range(args.n_layers)])
            rnn_idim = args.emb_dim + args.n_units_null_context
            for l in range(args.n_layers):
                self.rnn += [rnn(rnn_idim, args.n_units, 1,
                                 bias=True,
                                 batch_first=True,
                                 dropout=0,
                                 bidirectional=False)]
                rnn_idim = args.n_units
                if args.n_projs > 0:
                    rnn_idim = args.n_projs

        if self.use_glu:
            self.fc_glu = Linear(rnn_idim, rnn_idim * 2,
                                 dropout=args.dropout_hidden)

        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                rnn_idim, self.vocab,
                # cutoffs=[self.vocab // 10, 3 * self.vocab // 10],
                cutoffs=[self.vocab // 25, self.vocab // 5],
                div_value=4.0)
            self.output = None
        else:
            self.adaptive_softmax = None
            self.output = Linear(rnn_idim, self.vocab,
                                 dropout=args.dropout_out)
            # NOTE: include bias even when tying weights

            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            if args.tie_embedding:
                if args.n_units != args.emb_dim:
                    raise ValueError('When using the tied flag, n_units must be equal to emb_dim.')
                self.output.fc.weight = self.embed.embed.weight

        # Initialize parameters
        self.reset_parameters(args.param_init)

        # Recurrent weights are orthogonalized
        if args.rec_weight_orthogonal:
            self.reset_parameters(args.param_init, dist='orthogonal',
                                  keys=['rnn', 'weight'])

        # Initialize bias in forget gate with 1
        # self.init_forget_gate_bias_with_one()

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, val=0)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0))
            elif p.dim() == 2:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError

    def decode(self, ys, state, is_asr=False):
        """Decode function.

        Args:
            ys (FloatTensor): `[B, L]`
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
            is_asr (bool):
        Returns:
            ys_emb (FloatTensor): `[B, L, n_units]`
            new_state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        ys_emb = self.embed(ys.long())

        if state is None:
            state = self.zero_state(ys_emb.size(0))
        new_state = {'hxs': None, 'cxs': None}

        if self.n_units_cv > 0:
            cv = ys_emb.new_zeros(ys_emb.size(0), ys_emb.size(1), self.n_units_cv)
            ys_emb = torch.cat([ys_emb, cv], dim=-1)

        residual = None
        if self.fast_impl:
            # Path through RNN
            if self.rnn_type == 'lstm':
                ys_emb, (new_state['hxs'], new_state['cxs']) = self.rnn(
                    ys_emb, hx=(state['hxs'], state['cxs']))
            elif self.rnn_type == 'gru':
                ys_emb, new_state['hxs'] = self.rnn(ys_emb, hx=state['hxs'])
            ys_emb = self.dropout_top(ys_emb)
        else:
            new_hxs, new_cxs = [], []
            for l in range(self.n_layers):
                # Path through RNN
                if self.rnn_type == 'lstm':
                    ys_emb, (h_l, c_l) = self.rnn[l](ys_emb, hx=(state['hxs'][l:l + 1],
                                                                 state['cxs'][l:l + 1]))
                    new_cxs.append(c_l)
                elif self.rnn_type == 'gru':
                    ys_emb, h_l = self.rnn[l](ys_emb, hx=state['hxs'][l:l + 1])
                new_hxs.append(h_l)
                ys_emb = self.dropout[l](ys_emb)
                if self.n_projs > 0:
                    ys_emb = torch.tanh(self.proj[l](ys_emb))

                # Residual connection
                if self.residual and l > 0:
                    ys_emb = ys_emb + residual
                residual = ys_emb
                # NOTE: Exclude residual connection from the raw inputs

            # Repackage
            new_state['hxs'] = torch.cat(new_hxs, dim=0)
            if self.rnn_type == 'lstm':
                new_state['cxs'] = torch.cat(new_cxs, dim=0)

        if self.use_glu:
            if self.residual:
                residual = ys_emb
            ys_emb = F.glu(self.fc_glu(ys_emb), dim=-1)
            if self.residual:
                ys_emb = ys_emb + residual

        return ys_emb, new_state

    def zero_state(self, batch_size):
        """Initialize hidden states.

        Args:
            batch_size (int):
        Returns:
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        w = next(self.parameters())
        state = {'hxs': None, 'cxs': None}
        state['hxs'] = w.new_zeros(self.n_layers, batch_size, self.n_units)
        if self.rnn_type == 'lstm':
            state['cxs'] = w.new_zeros(self.n_layers, batch_size, self.n_units)
        return state

    def repackage_state(self, state):
        """Wraps hidden states in new Tensors, to detach them from their history.

        Args:
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
        Returns:
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        state['hxs'] = state['hxs'].detach()
        if self.rnn_type == 'lstm':
            state['cxs'] = state['cxs'].detach()
        return state

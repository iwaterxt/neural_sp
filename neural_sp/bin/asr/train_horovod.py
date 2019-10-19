#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Train the ASR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import cProfile
# import editdistance
import numpy as np
import os
from setproctitle import setproctitle
import shutil
import time
import torch
import horovod.torch as hvd
from tqdm import tqdm

from neural_sp.bin.args_asr import parse
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import save_checkpoint
from neural_sp.bin.train_utils import save_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.bin.train_utils import set_save_path
from neural_sp.datasets.asr_parallel import Dataset
from neural_sp.evaluators.character import eval_char
from neural_sp.evaluators.phone import eval_phone
from neural_sp.evaluators.ppl import eval_ppl_parallel
from neural_sp.evaluators.word import eval_word
from neural_sp.evaluators.wordpiece import eval_wordpiece
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.models.lm.build import build_lm
from neural_sp.models.seq2seq.speech2text import Speech2Text
from neural_sp.trainers.lr_scheduler import LRScheduler
from neural_sp.trainers.model_name import set_asr_model_name
from neural_sp.trainers.optimizer import set_optimizer
from neural_sp.trainers.reporter import Reporter
from neural_sp.utils import mkdir_join
from neural_sp.utils import host_ip
from neural_sp.models.torch_utils import tensor2np
from neural_sp.models.torch_utils import np2tensor
from neural_sp.bin.seqdataloader import SeqDataloader

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def main():

    args = parse()
    args_pt = copy.deepcopy(args)


    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    hvd_rank = hvd.rank()
    # Load a conf file
    if args.resume:
        conf = load_config(os.path.join(os.path.dirname(args.resume), 'conf.yml'))
        for k, v in conf.items():
            if k != 'resume':
                setattr(args, k, v)
    recog_params = vars(args)

    # Compute subsampling factor
    subsample_factor = 1

    subsample = [int(s) for s in args.subsample.split('_')]
    if args.conv_poolings and 'conv' in args.enc_type:
        for p in args.conv_poolings.split('_'):
            subsample_factor *= int(p.split(',')[0].replace('(', ''))
    else:
        subsample_factor = np.prod(subsample)

    skip_thought = 'skip' in args.enc_type
    batch_per_allreduce = args.batch_size 
    # Load dataset
    train_set = Dataset(corpus=args.corpus,
                        tsv_path=args.train_set,
                        tsv_path_sub1=args.train_set_sub1,
                        tsv_path_sub2=args.train_set_sub2,
                        dict_path=args.dict,
                        dict_path_sub1=args.dict_sub1,
                        dict_path_sub2=args.dict_sub2,
                        nlsyms=args.nlsyms,
                        unit=args.unit,
                        unit_sub1=args.unit_sub1,
                        unit_sub2=args.unit_sub2,
                        wp_model=args.wp_model,
                        wp_model_sub1=args.wp_model_sub1,
                        wp_model_sub2=args.wp_model_sub2,
                        batch_size=args.batch_size,
                        n_epochs=args.n_epochs,
                        min_n_frames=args.min_n_frames,
                        max_n_frames=args.max_n_frames,
                        sort_by='no_sort',
                        short2long=True,
                        sort_stop_epoch=args.sort_stop_epoch,
                        dynamic_batching=args.dynamic_batching,
                        ctc=args.ctc_weight > 0,
                        ctc_sub1=args.ctc_weight_sub1 > 0,
                        ctc_sub2=args.ctc_weight_sub2 > 0,
                        subsample_factor=subsample_factor,
                        discourse_aware=args.discourse_aware,
                        skip_thought=skip_thought)

    dev_set = Dataset(corpus=args.corpus,
                      tsv_path=args.dev_set,
                      tsv_path_sub1=args.dev_set_sub1,
                      tsv_path_sub2=args.dev_set_sub2,
                      dict_path=args.dict,
                      dict_path_sub1=args.dict_sub1,
                      dict_path_sub2=args.dict_sub2,
                      nlsyms=args.nlsyms,
                      unit=args.unit,
                      unit_sub1=args.unit_sub1,
                      unit_sub2=args.unit_sub2,
                      wp_model=args.wp_model,
                      wp_model_sub1=args.wp_model_sub1,
                      wp_model_sub2=args.wp_model_sub2,
                      batch_size=args.batch_size,
                      min_n_frames=args.min_n_frames,
                      max_n_frames=args.max_n_frames,
                      ctc=args.ctc_weight > 0,
                      ctc_sub1=args.ctc_weight_sub1 > 0,
                      ctc_sub2=args.ctc_weight_sub2 > 0,
                      subsample_factor=subsample_factor,
                      discourse_aware=args.discourse_aware,
                      skip_thought=skip_thought)
    eval_sets = []
    for s in args.eval_sets:
        eval_sets += [Dataset(corpus=args.corpus,
                              tsv_path=s,
                              dict_path=args.dict,
                              nlsyms=args.nlsyms,
                              unit=args.unit,
                              wp_model=args.wp_model,
                              batch_size=1,
                              discourse_aware=args.discourse_aware,
                              skip_thought=skip_thought,
                              is_test=True)]

    args.vocab = train_set.vocab
    args.vocab_sub1 = train_set.vocab_sub1
    args.vocab_sub2 = train_set.vocab_sub2
    args.input_dim = train_set.input_dim
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_loader = SeqDataloader(train_set, 
                                 batch_size=args.batch_size,
                                 num_workers = 1,
                                 distributed=True,
                                 num_stacks=args.n_stacks,
                                 num_splices=args.n_splices,
                                 num_skips=args.n_skips,
                                 pin_memory=False
                                )
    val_loader = SeqDataloader(dev_set, 
                               batch_size=args.batch_size,
                               num_workers = 1,
                               distributed=True,
                               num_stacks=args.n_stacks,
                               num_splices=args.n_splices,
                               num_skips=args.n_skips,
                               pin_memory=False
                              )    

    # Load a LM conf file for LM fusion & LM initialization
    if not args.resume and (args.lm_fusion or args.lm_init):
        if args.lm_fusion:
            lm_conf = load_config(os.path.join(os.path.dirname(args.lm_fusion), 'conf.yml'))
        elif args.lm_init:
            lm_conf = load_config(os.path.join(os.path.dirname(args.lm_init), 'conf.yml'))
        args.lm_conf = argparse.Namespace()
        for k, v in lm_conf.items():
            setattr(args.lm_conf, k, v)
        assert args.unit == args.lm_conf.unit
        assert args.vocab == args.lm_conf.vocab

    # Set save path
    if args.resume:
        save_path = os.path.dirname(args.resume)
        dir_name = os.path.basename(save_path)
    else:
        dir_name = set_asr_model_name(args, subsample_factor)
        save_path = mkdir_join(args.model_save_dir, '_'.join(
            os.path.basename(args.train_set).split('.')[:-1]), dir_name)
        if hvd.rank() == 0:
            save_path = set_save_path(save_path)  # avoid overwriting

    # Set logger
    if hvd_rank == 0:
        logger = set_logger(os.path.join(save_path, 'train.log'),
                            key='training', stdout=args.stdout)
        # Set process name
        logger.info('PID: %s' % os.getpid())
        logger.info('USERNAME: %s' % os.uname()[1])
        logger.info('NUMBER_DEVICES: %s' % hvd.size())
    
    setproctitle(args.job_name if args.job_name else dir_name)
    # Model setting
    model = Speech2Text(args, save_path) 
    # GPU setting
    if args.n_gpus >= 1:
        torch.backends.cudnn.benchmark = True
        model.cuda()

    if args.resume :
        # Set optimizer
        epochs = int(args.resume.split('-')[-1])
        #optimizer = set_optimizer(model, 'sgd' if epochs >= conf['convert_to_sgd_epoch'] else conf['optimizer'],

        model, _ = load_checkpoint(model, args.resume, resume=True)
        optimizer = set_optimizer(model, 'sgd',conf['lr'], conf['weight_decay'])
        #broadcast
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())




        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # Wrap optimizer by learning rate scheduler
        noam = 'transformer' in args.enc_type or args.dec_type == 'transformer'
        optimizer = LRScheduler(optimizer, args.lr,
                                decay_type=args.lr_decay_type,
                                decay_start_epoch=args.lr_decay_start_epoch,
                                decay_rate=args.lr_decay_rate,
                                decay_patient_n_epochs=args.lr_decay_patient_n_epochs,
                                early_stop_patient_n_epochs=args.early_stop_patient_n_epochs,
                                warmup_start_lr=args.warmup_start_lr,
                                warmup_n_steps=args.warmup_n_steps,
                                model_size=args.d_model,
                                factor=args.lr_factor,
                                noam=noam)

    else:
        # Save the conf file as a yaml file
        if hvd_rank == 0:
            save_config(vars(args), os.path.join(save_path, 'conf.yml'))
        if args.lm_fusion:
            save_config(args.lm_conf, os.path.join(save_path, 'conf_lm.yml'))

        # Save the nlsyms, dictionar, and wp_model
        if args.nlsyms:
            shutil.copy(args.nlsyms, os.path.join(save_path, 'nlsyms.txt'))

        if hvd_rank == 0:
            for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
                logger.info('%s: %s' % (k, str(v)))

            # Count total parameters
            for n in sorted(list(model.num_params_dict.keys())):
                n_params = model.num_params_dict[n]
                logger.info("%s %d" % (n, n_params))
            logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))
            logger.info(model)

        # Set optimizer
        optimizer = set_optimizer(model, args.optimizer, args.lr, args.weight_decay)

        #compression = hvd.Compression.fp16 if args.f16_allreduce else hvd.Compression.none

        optimizer = hvd.DistributedOptimizer(
                        optimizer, named_parameters=model.named_parameters(),
                        compression=hvd.Compression.none,
                        backward_passes_per_step=batch_per_allreduce)


        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # Wrap optimizer by learning rate scheduler
        noam = 'transformer' in args.enc_type or args.dec_type == 'transformer'
        optimizer = LRScheduler(optimizer, args.lr,
                                decay_type=args.lr_decay_type,
                                decay_start_epoch=args.lr_decay_start_epoch,
                                decay_rate=args.lr_decay_rate,
                                decay_patient_n_epochs=args.lr_decay_patient_n_epochs,
                                early_stop_patient_n_epochs=args.early_stop_patient_n_epochs,
                                warmup_start_lr=args.warmup_start_lr,
                                warmup_n_steps=args.warmup_n_steps,
                                model_size=args.d_model,
                                factor=args.lr_factor,
                                noam=noam)
    # Set reporter
    reporter = Reporter(save_path)
    if args.mtl_per_batch:
        # NOTE: from easier to harder tasks
        tasks = []
        if 1 - args.bwd_weight - args.ctc_weight - args.sub1_weight - args.sub2_weight > 0:
            tasks += ['ys']
        if args.bwd_weight > 0:
            tasks = ['ys.bwd'] + tasks
        if args.ctc_weight > 0:
            tasks = ['ys.ctc'] + tasks
    else:
        tasks = ['all']

    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    accum_n_tokens = 0

    verbose = 1 if hvd_rank == 0 else 0
    data_size = len(train_set)
    while True:
      model.train()
      with tqdm(total=data_size//hvd.size(),
              desc='Train Epoch     #{}'.format(optimizer.n_epochs + 1),
              disable=not verbose) as pbar_epoch:
        # Compute loss in the training set
        for _, batch_train in enumerate(train_loader):
            accum_n_tokens += sum([len(y) for y in batch_train['ys']])
            # Change mini-batch depending on task
            for task in tasks:
                if skip_thought:
                    loss, reporter = model(batch_train['ys'],
                                       ys_prev=batch_train['ys_prev'],
                                       ys_next=batch_train['ys_next'],
                                       reporter=reporter)
                else:
                    loss, reporter = model(batch_train, reporter, task)
                loss.backward()
                loss.detach()  # Trancate the graph
                if args.accum_grad_n_tokens == 0 or accum_n_tokens >= args.accum_grad_n_tokens:
                    if args.clip_grad_norm > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.clip_grad_norm)
                        reporter.add_tensorboard_scalar('total_norm', total_norm)
                    optimizer.step()
                    
                    optimizer.zero_grad()

                    accum_n_tokens = 0
                loss_train = loss.item()
                del loss
            if hvd_rank == 0:
                reporter.add_tensorboard_scalar('learning_rate', optimizer.lr)
                # NOTE: loss/acc/ppl are already added in the model
                reporter.step()

            if optimizer.n_steps % args.print_step == 0:
                # Compute loss in the dev set
                model.eval()
                batch_dev = dev_set.next()[0]
                # Change mini-batch depending on task
                for task in tasks:
                    if skip_thought:
                        loss, reporter = model(batch_dev['ys'],
                                           ys_prev=batch_dev['ys_prev'],
                                           ys_next=batch_dev['ys_next'],
                                           reporter=reporter,
                                           is_eval=True)
                    else:
                        loss, reporter = model(batch_dev, reporter, task, is_eval=True)
                    loss_dev = loss.item()
                    del loss

                

                duration_step = time.time() - start_time_step
                if args.input_type == 'speech':
                    xlen = max(len(x) for x in batch_train['xs'])
                    ylen = max(len(y) for y in batch_train['ys'])
                elif args.input_type == 'text':
                    xlen = max(len(x) for x in batch_train['ys'])
                    ylen = max(len(y) for y in batch_train['ys_sub1'])

                if hvd_rank == 0:
                    reporter.step(is_eval=True)
                    logger.info("step:%d(ep:%.2f) loss:%.3f(%.3f)/lr:%.5f/bs:%d/xlen:%d/ylen:%d (%.2f min)" %
                                (optimizer.n_steps, optimizer.n_steps*args.batch_size/(data_size/hvd.size()),
                                loss_train, loss_dev,
                                optimizer.lr, len(batch_train['utt_ids']),
                                xlen, ylen, duration_step / 60))
                start_time_step = time.time()
            pbar_epoch.update(len(batch_train['utt_ids']))

            # Save fugures of loss and accuracy
            if optimizer.n_steps % (args.print_step * 10) == 0 and hvd.rank() == 0:
                reporter.snapshot()
                model.plot_attention()
            start_time_step = time.time()
        # Save checkpoint and evaluate model per epoch
        
        duration_epoch = time.time() - start_time_epoch
        if hvd_rank == 0:
            logger.info('========== EPOCH:%d (%.2f min) ==========' %
                        (optimizer.n_epochs + 1, duration_epoch / 60))

        if optimizer.n_epochs + 1 < args.eval_start_epoch:
            optimizer.epoch()
            if hvd_rank == 0:
                reporter.epoch()
                save_checkpoint(model, save_path, optimizer, optimizer.n_epochs,
                                    remove_old_checkpoints=not noam)
        else:
            start_time_eval = time.time()
            # dev

            metric_dev = eval_epoch([model], val_loader, recog_params, args, optimizer.n_epochs + 1)

            metric_dev = hvd.allreduce(np2tensor(np.array([metric_dev], dtype=float), hvd.local_rank()))
            loss_dev = metric_dev.item()
            if hvd_rank == 0:
                logger.info('Loss : %.2f %%' % (loss_dev))
                reporter.epoch(loss_dev)
            optimizer.epoch(loss_dev)
            if hvd.rank() == 0:
                if optimizer.is_best :
                    # Save the model
                    save_checkpoint(model, save_path, optimizer, optimizer.n_epochs,
                                        remove_old_checkpoints=not noam)
                else:
                    # Load best model
                    model, _ = load_checkpoint(model, save_path+'/model.epoch-'+str(optimizer.best_epochs))
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                # start scheduled sampling
            if args.ss_prob > 0:
                model.scheduled_sampling_trigger()

            duration_eval = time.time() - start_time_eval
            if hvd_rank == 0:
                logger.info('Evaluation time: %.2f min' % (duration_eval / 60))

            # Early stopping
            if optimizer.is_early_stop:
                break
        # Convert to fine-tuning stage
        if optimizer.n_epochs == args.convert_to_sgd_epoch:
            n_epochs = optimizer.n_epochs
            n_steps = optimizer.n_steps
            optimizer = set_optimizer(model, 'sgd', args.lr, args.weight_decay)
            optimizer = hvd.DistributedOptimizer(
                            optimizer, named_parameters=model.named_parameters(),
                            compression=hvd.Compression.none,
                            backward_passes_per_step=batch_per_allreduce)


            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            optimizer = LRScheduler(optimizer, args.lr,
                                    decay_type='always',
                                    decay_start_epoch=0,
                                    decay_rate=0.5)

            optimizer._epoch = n_epochs
            optimizer._step = n_steps
            if hvd_rank == 0:
                logger.info('========== Convert to SGD ==========')


        if optimizer.n_epochs == args.n_epochs:
            break
        start_time_step = time.time()
        start_time_epoch = time.time()

    duration_train = time.time() - start_time_train
    if hvd_rank == 0:
        logger.info('Total time: %.2f hour' % (duration_train / 3600))

    reporter.tf_writer.close()
    #pbar_epoch.close()

    return save_path


def eval_epoch(models, dataloader, recog_params, args, epochs):
    if args.metric == 'loss':
        metric = eval_ppl_parallel(models, dataloader, epochs, batch_size=args.batch_size)[1]
    else:
        raise NotImplementedError(args.metric)
    return metric


if __name__ == '__main__':
    # Setting for profiling

    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))

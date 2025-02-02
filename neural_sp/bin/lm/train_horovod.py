#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Train the LM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
import numpy as np
import os
from setproctitle import setproctitle
import shutil
import time
import torch
from tqdm import tqdm
import horovod.torch as hvd

from neural_sp.bin.args_lm import parse
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import save_checkpoint
from neural_sp.bin.train_utils import save_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.bin.train_utils import set_save_path
from neural_sp.datasets.lm_parallel import Dataset
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.evaluators.ppl import eval_ppl_parallel
from neural_sp.models.data_parallel import CustomDataParallel
from neural_sp.bin.seqdataloader import ChunkDataloader
from neural_sp.models.lm.build import build_lm
from neural_sp.trainers.lr_scheduler import LRScheduler
from neural_sp.trainers.model_name import set_lm_name
from neural_sp.trainers.optimizer import set_optimizer
from neural_sp.trainers.reporter import Reporter
from neural_sp.models.torch_utils import np2tensor
from neural_sp.utils import mkdir_join

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def main():

    args = parse()

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    hvd_rank = hvd.rank()
    # Load a conf file
    if args.resume:
        conf = load_config(os.path.join(os.path.dirname(args.resume), 'conf.yml'))
        for k, v in conf.items():
            if k != 'resume':
                setattr(args, k, v)

    # Load dataset
    train_set = Dataset(corpus=args.corpus,
                        tsv_path=args.train_set,
                        dict_path=args.dict,
                        nlsyms=args.nlsyms,
                        unit=args.unit,
                        wp_model=args.wp_model,
                        batch_size=args.batch_size,
                        n_epochs=args.n_epochs,
                        min_n_tokens=args.min_n_tokens,
                        bptt=args.bptt,
                        n_customers=hvd.size(),
                        backward=args.backward,
                        serialize=args.serialize)
    dev_set = Dataset(corpus=args.corpus,
                      tsv_path=args.dev_set,
                      dict_path=args.dict,
                      nlsyms=args.nlsyms,
                      unit=args.unit,
                      wp_model=args.wp_model,
                      batch_size=args.batch_size,
                      bptt=args.bptt,
                      n_customers=hvd.size(),
                      backward=args.backward,
                      serialize=args.serialize)

    eval_set = Dataset(corpus=args.corpus,
                              tsv_path=args.eval_set,
                              dict_path=args.dict,
                              nlsyms=args.nlsyms,
                              unit=args.unit,
                              wp_model=args.wp_model,
                              batch_size=args.batch_size,
                              bptt=args.bptt,
                              n_customers=hvd.size(),
                              backward=args.backward,
                              serialize=args.serialize)

    args.vocab = train_set.vocab

    train_loader = ChunkDataloader(train_set,
                                   batch_size=1,
                                   num_workers = 1,
                                   distributed=True,
                                   shuffle=False)

    eval_loader = ChunkDataloader(eval_set,
                                 batch_size=1,
                                 num_workers=1,
                                 distributed=True)




    # Set save path
    if args.resume:
        save_path = os.path.dirname(args.resume)
        dir_name = os.path.basename(save_path)
    else:
        dir_name = set_lm_name(args)
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
    model = build_lm(args, save_path)
    # GPU setting
    if args.n_gpus >= 1:
        torch.backends.cudnn.benchmark = True
        model.cuda()

    if args.resume:
        # Set optimizer
        epoch = int(args.resume.split('-')[-1])
        optimizer = set_optimizer(model, 'sgd' if epoch > conf['convert_to_sgd_epoch'] else conf['optimizer'],
                                  conf['lr'], conf['weight_decay'])

        # Restore the last saved model
        if hvd_rank == 0:
            model, optimizer = load_checkpoint(model, args.resume, optimizer, resume=True)
        #broadcast
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # Wrap optimizer by learning rate scheduler
        optimizer = LRScheduler(optimizer, conf['lr'],
                                decay_type=conf['lr_decay_type'],
                                decay_start_epoch=conf['lr_decay_start_epoch'],
                                decay_rate=conf['lr_decay_rate'],
                                decay_patient_n_epochs=conf['lr_decay_patient_n_epochs'],
                                early_stop_patient_n_epochs=conf['early_stop_patient_n_epochs'],
                                warmup_start_lr=conf['warmup_start_lr'],
                                warmup_n_steps=conf['warmup_n_steps'],
                                model_size=conf['d_model'],
                                factor=conf['lr_factor'],
                                noam=conf['lm_type'] == 'transformer')

        # Resume between convert_to_sgd_epoch -1 and convert_to_sgd_epoch
        if epoch == conf['convert_to_sgd_epoch']:
            n_epochs = optimizer.n_epochs
            n_steps = optimizer.n_steps
            optimizer = set_optimizer(model, 'sgd', args.lr, conf['weight_decay'])
            optimizer = LRScheduler(optimizer, args.lr,
                                    decay_type='always',
                                    decay_start_epoch=0,
                                    decay_rate=0.5)
            optimizer._epoch = n_epochs
            optimizer._step = n_steps
            if hvd_rank == 0:
                logger.info('========== Convert to SGD ==========')
            #broadcast
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    else:
        # Save the conf file as a yaml file
        if hvd_rank == 0:
            save_config(vars(args), os.path.join(save_path, 'conf.yml'))
            # Save the nlsyms, dictionar, and wp_model
            if args.nlsyms:
                shutil.copy(args.nlsyms, os.path.join(save_path, 'nlsyms.txt'))
            shutil.copy(args.dict, os.path.join(save_path, 'dict.txt'))
            if args.unit == 'wp':
                shutil.copy(args.wp_model, os.path.join(save_path, 'wp.model'))
            for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
                logger.info('%s: %s' % (k, str(v)))

        # Count total parameters
        for n in sorted(list(model.num_params_dict.keys())):
            n_params = model.num_params_dict[n]
            if hvd.rank() == 0:
                logger.info("%s %d" % (n, n_params))
        if hvd_rank == 0:
            logger.info("Total %.2f M parameters" % (model.total_parameters / 1000000))
            logger.info(model)

        # Set optimizer
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        optimizer = set_optimizer(model, args.optimizer, args.lr, args.weight_decay)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # Wrap optimizer by learning rate scheduler
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
                                noam=args.lm_type == 'transformer')

    

    # Set reporter
    reporter = Reporter(save_path)

    hidden = None
    start_time_train = time.time()
    start_time_epoch = time.time()
    start_time_step = time.time()
    data_size = len(train_set)
    accum_n_tokens = 0
    verbose = 1 if hvd_rank == 0 else 0
    while True:
        model.train()
        with tqdm(total=data_size/hvd.size(),
                desc='Train Epoch     #{}'.format(optimizer.n_epochs + 1),
                disable=not verbose) as pbar_epoch:
            # Compute loss in the training set
            for _, ys_train in enumerate(train_loader):
                accum_n_tokens += sum([len(y) for y in ys_train])
                optimizer.zero_grad()
                loss, hidden, reporter = model(ys_train, hidden, reporter)
                loss.backward()
                loss.detach()  # Trancate the graph
                if args.accum_grad_n_tokens == 0 or accum_n_tokens >= args.accum_grad_n_tokens:
                    if args.clip_grad_norm > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.clip_grad_norm)
                        #reporter.add_tensorboard_scalar('total_norm', total_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_n_tokens = 0
                loss_train = loss.item()
                del loss
                hidden = model.repackage_state(hidden)
                
                if optimizer.n_steps % args.print_step == 0:
                    model.eval()
                    # Compute loss in the dev set
                    ys_dev = dev_set.next()[0]
                    loss, _, reporter = model(ys_dev, None, reporter, is_eval=True)
                    loss_dev = loss.item()
                    del loss
                    
                    duration_step = time.time() - start_time_step
                    if hvd_rank == 0:
                    	logger.info("step:%d(ep:%.2f) loss:%.3f(%.3f)/ppl:%.3f(%.3f)/lr:%.5f/bs:%d (%.2f min)" %
                                    (optimizer.n_steps, optimizer.n_steps/data_size*hvd.size(),
                                    loss_train, loss_dev,
                                    np.exp(loss_train), np.exp(loss_dev),
                                    optimizer.lr, ys_train.shape[0], duration_step / 60))
                    start_time_step = time.time()
                
                pbar_epoch.update(1)
                

            # Save checkpoint and evaluate model per epoch
            duration_epoch = time.time() - start_time_epoch
            if hvd_rank == 0:
                logger.info('========== EPOCH:%d (%.2f min) ==========' %(optimizer.n_epochs + 1, duration_epoch / 60))

            if optimizer.n_epochs + 1 < args.eval_start_epoch:

                # Save the model
                if hvd_rank == 0:
                    optimizer.epoch()
                    save_checkpoint(model, save_path, optimizer, optimizer.n_epochs,
                                        remove_old_checkpoints=args.lm_type != 'transformer')
            else:
                start_time_eval = time.time()
                # dev
                model.eval()
                ppl_dev, _ = eval_ppl_parallel([model], eval_loader, optimizer.n_epochs, batch_size=args.batch_size)
                ppl_dev = hvd.allreduce(np2tensor(np.array([ppl_dev], dtype=float), hvd.local_rank()))
                
                if hvd_rank == 0:
                    logger.info('PPL : %.2f' %  ppl_dev)
                optimizer.epoch(ppl_dev)

                if optimizer.is_best and hvd.rank() == 0:
                    # Save the model
                    save_checkpoint(model, save_path, optimizer, optimizer.n_epochs,
                                    remove_old_checkpoints=args.lm_type != 'transformer')

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
                                    optimizer, named_parameters=model.named_parameters())
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

    return save_path


if __name__ == '__main__':
    # Setting for profiling
    pr = cProfile.Profile()
    save_path = pr.runcall(main)
    pr.dump_stats(os.path.join(save_path, 'train.profile'))

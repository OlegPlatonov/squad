import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAFGT
from tensorboardX import SummaryWriter
from tqdm import tqdm
from util import gapped_text_collate_fn, GappedText

from apex.parallel import DistributedDataParallel as DDP


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    if args.local_rank == 0:
        tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))

    # Set random seed
    log.info('Using random seed {}...'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    char_vectors = util.torch_from_json(args.char_emb_file)

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')

    log.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))


    # Get model
    log.info('Building model...')
    model = BiDAFGT(word_vectors=word_vectors,
                    char_vectors=char_vectors,
                    hidden_size=args.hidden_size,
                    hidden_size_2=args.hidden_size_2,
                    drop_prob=args.drop_prob)

    log.info('Loading model...')
    ckpt_dict = torch.load(args.load_path)
    model.load_state_dict(ckpt_dict)

    log.info('Encoder:')
    log.info(model.encoder)
    log.info('Output_layer:')
    log.info(model.output_layer)

    model = model.to(device)
    model = DDP(model)
    model.train()
    step = 0

    # Get saver
    if args.local_rank == 0:
        saver = util.CheckpointSaver(args.save_dir,
                                     max_checkpoints=args.max_checkpoints,
                                     metric_name='Accuracy',
                                     maximize_metric=args.maximize_metric,
                                     log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.l2_wd)
    optimizer.defaults['lr'] = args.lr
    log.info(f'Optimizer: {optimizer}')
    log.info(f'Default learning rate is set to {optimizer.defaults["lr"]}')
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Creating train dataset...')
    train_file = './data/Pretraining/Dataset_1.npz'
    train_dataset = GappedText(train_file)
    train_sampler = data.distributed.DistributedSampler(train_dataset)
    train_loader = data.DataLoader(train_dataset,
                                   sampler=train_sampler,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=4,
                                   collate_fn=gapped_text_collate_fn)

    log.info('Creating dev dataset...')
    dev_file = './data/Pretraining/Dataset_dev.npz'
    dev_dataset = GappedText(dev_file)
    dev_sampler = data.distributed.DistributedSampler(dev_dataset)
    dev_loader = data.DataLoader(dev_dataset,
                                 sampler=dev_sampler,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=gapped_text_collate_fn)

    train_dataset_size = len(train_loader.dataset)
    num_train_optimization_steps = int(train_dataset_size / args.batch_size)
    log.info(f'Number of optimizations steps: {num_train_optimization_steps}')
    num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    log.info(f'Number of optimizations steps: {num_train_optimization_steps}')

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // train_dataset_size
    while epoch != args.num_epochs:
        model.train()
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        with torch.enable_grad(), \
                tqdm(total=train_dataset_size) as progress_bar:
            for cw_idxs, cc_idxs, gap_indices, qw_idxs, qc_idxs, correct_gaps in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                cc_idxs = cc_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
                gap_indices = gap_indices.to(device)
                correct_gaps = correct_gaps.to(device)
                batch_size = int(cw_idxs.size(0) / args.num_fragments)
                optimizer.zero_grad()

                # Forward
                loss = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs, gap_indices, correct_gaps=correct_gaps)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                if args.local_rank == 0:
                    tbx.add_scalar('train/NLL', loss_val, step)
                    tbx.add_scalar('train/LR',
                                   optimizer.param_groups[0]['lr'],
                                   step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info('Saving checkpoint at step {}...'.format(step))
                    results = evaluate(model, dev_loader, device, args)
                    if args.local_rank == 0:
                        saver.save(step, model, optimizer, results['Accuracy'], device)

                    # Log to console
                    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                            for k, v in results.items())
                    log.info('Dev {}'.format(results_str))

                    # Log to TensorBoard
                    if args.local_rank == 0:
                        log.info('Visualizing in TensorBoard...')
                        for k, v in results.items():
                            tbx.add_scalar('dev/{}'.format(k), v, step)

            if args.eval_after_epoch:
                # Evaluate and save checkpoint
                log.info('Saving checkpoint at step {}...'.format(step))
                results = evaluate(model, dev_loader, device, args)
                if args.local_rank == 0:
                    saver.save(step, model, optimizer, results['Accuracy'], device)

                # Log to console
                results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                        for k, v in results.items())
                log.info('Dev {}'.format(results_str))

                # Log to TensorBoard
                if args.local_rank == 0:
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar('dev/{}'.format(k), v, step)


def evaluate(model, data_loader, device, args):
    nll_meter = util.AverageMeter()

    model.eval()
    correct_preds = 0
    correct_avna = 0
    zero_preds = 0
    total_preds = 0
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, gap_indices, qw_idxs, qc_idxs, correct_gaps in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            gap_indices = gap_indices.to(device)
            correct_gaps = correct_gaps.to(device)
            batch_size = int(cw_idxs.size(0) / args.num_fragments)

            # Forward
            logits = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs, gap_indices)
            loss = F.cross_entropy(input=logits, target=correct_gaps)
            nll_meter.update(loss.item(), batch_size)

            preds = torch.argmax(logits, dim=1)
            correct_preds += torch.sum(preds == correct_gaps).item()
            correct_avna += torch.sum((preds > 0) == (correct_gaps > 0)).item()
            zero_preds += torch.sum(preds == 0).item()
            total_preds += preds.shape[0]

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

    model.train()

    results_list = [('NLL', nll_meter.avg),
                    ('Accuracy', correct_preds / total_preds),
                    ('AvNA', correct_avna / total_preds),
                    ('NA_share', zero_preds / total_preds)]

    results = OrderedDict(results_list)

    return results


if __name__ == '__main__':
    main(get_train_args())

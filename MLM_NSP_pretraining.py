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
from models import BiDAFMLMNSP
from tensorboardX import SummaryWriter
from tqdm import tqdm
from util import MLM_NSP_collate_fn, MLM_NSP
from functools import partial


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    args.batch_size *= max(1, len(args.gpu_ids))

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

    # Get model
    log.info('Building model...')
    model = BiDAFMLMNSP(word_vectors=word_vectors,
                        char_vectors=char_vectors,
                        hidden_size=args.hidden_size,
                        hidden_size_2=args.hidden_size_2,
                        drop_prob=args.drop_prob)

    log.info('Encoder:')
    log.info(model.encoder)
    log.info('Output_layer:')
    log.info(model.output_layer)

    model = nn.DataParallel(model, args.gpu_ids)

    if args.load_path:
        log.info('Loading model checkpoint from {}...'.format(args.load_path))
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name='Accuracy',
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.l2_wd)
    if args.load_path:
        log.info('Loading optimizer checkpoint from {}...'.format(args.load_path + '.optim'))
        optimizer.load_state_dict(torch.load(args.load_path + '.optim'))
    optimizer.defaults['lr'] = args.lr
    log.info(f'Optimizer: {optimizer}')
    log.info(f'Default learning rate is set to {optimizer.defaults["lr"]}')
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    data_folder = './data/MLM_NSP/Tokenized'
    data_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder)]
    log.info('Training data files found:')
    for file in data_files:
        log.info(file)

    log.info('Creating dev dataset...')
    dev_file = './data/MLM_NSP/Tokenized_dev/Dataset_dev.npz'
    dev_dataset = MLM_NSP(dev_file)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=partial(MLM_NSP_collate_fn,
                                                    use_pseudomasking=True,
                                                    vocab_size=word_vectors.shape[0]))

    # Train
    log.info('Training...')
    train_dataset_size = 5447847
    steps_till_eval = args.eval_steps
    epoch = step // train_dataset_size
    while epoch != args.num_epochs:
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        with torch.enable_grad(), \
                tqdm(total=train_dataset_size) as progress_bar:

            random.shuffle(data_files)
            datasets_start = 0
            while datasets_start < len(data_files):
                log.info('Building dataset...')
                datasets = []
                for file in data_files[datasets_start:datasets_start + 4]:
                    log.info(f'Creating dataset from {file}...')
                    datasets.append(MLM_NSP(file))
                log.info('Concatenating datasets...')
                train_dataset = data.ConcatDataset(datasets)
                train_loader = data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               collate_fn=partial(MLM_NSP_collate_fn,
                                                                  use_pseudomasking=True,
                                                                  vocab_size=word_vectors.shape[0]))

                for cw_idxs, cc_idxs, qw_idxs, qc_idxs, is_next, mask_1, masked_words_1, mask_2, masked_words_2 in train_loader:
                    # Setup for forward
                    cw_idxs = cw_idxs.to(device)
                    cc_idxs = cc_idxs.to(device)
                    qw_idxs = qw_idxs.to(device)
                    qc_idxs = qc_idxs.to(device)
                    is_next = is_next.to(device)
                    mask_1 = tuple(x.to(device) for x in mask_1)
                    mask_2 = tuple(x.to(device) for x in mask_2)
                    masked_words_1 = masked_words_1.to(device)
                    masked_words_2 = masked_words_2.to(device)
                    batch_size = cw_idxs.size(0)
                    optimizer.zero_grad()

                    # Forward
                    MLM_logits_1, MLM_logits_2, NSP_logits = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs, mask_1, mask_2)
                    MLM_logits = torch.cat((MLM_logits_1, MLM_logits_2), dim=0)
                    masked_words = torch.cat((masked_words_1, masked_words_2), dim=0)

                    MLM_loss = F.cross_entropy(input=MLM_logits, target=masked_words)
                    NSP_loss = args.NSP_weight * F.binary_cross_entropy_with_logits(input=NSP_logits, target=is_next.float())
                    loss = MLM_loss + NSP_loss

                    MLM_loss_val = MLM_loss.item()
                    NSP_loss_val = NSP_loss.item()
                    loss_val = loss.item()

                    # Backward
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step(step // batch_size)
                    ema(model, step // batch_size)

                    # Log info
                    step += batch_size
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(epoch=epoch,
                                             MLM_NLL=MLM_loss_val,
                                             NSP_NLL=NSP_loss_val,
                                             NLL=loss_val)
                    tbx.add_scalar('train/NLL', loss_val, step)
                    tbx.add_scalar('train/MLM_NLL', MLM_loss_val, step)
                    tbx.add_scalar('train/NSP_NLL', NSP_loss_val, step)
                    tbx.add_scalar('train/LR',
                                   optimizer.param_groups[0]['lr'],
                                   step)

                    steps_till_eval -= batch_size
                    if steps_till_eval <= 0:
                        steps_till_eval = args.eval_steps

                        # Evaluate and save checkpoint
                        log.info('Saving checkpoint at step {}...'.format(step))
                        ema.assign(model)
                        results = evaluate(model, dev_loader, device, args)
                        saver.save(step, model, optimizer, results['Accuracy'], device)
                        ema.resume(model)

                        # Log to console
                        results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                                for k, v in results.items())
                        log.info('Dev {}'.format(results_str))

                        # Log to TensorBoard
                        log.info('Visualizing in TensorBoard...')
                        for k, v in results.items():
                            tbx.add_scalar('dev/{}'.format(k), v, step)

                datasets_start += 4
                del datasets

            if args.eval_after_epoch:
                # Evaluate and save checkpoint
                log.info('Saving checkpoint at step {}...'.format(step))
                ema.assign(model)
                results = evaluate(model, dev_loader, device, args)
                saver.save(step, model, optimizer, results['Accuracy'], device)
                ema.resume(model)

                # Log to console
                results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                        for k, v in results.items())
                log.info('Dev {}'.format(results_str))

                # Log to TensorBoard
                log.info('Visualizing in TensorBoard...')
                for k, v in results.items():
                    tbx.add_scalar('dev/{}'.format(k), v, step)


def evaluate(model, data_loader, device, args):
    nll_meter = util.AverageMeter()
    MLM_nll_meter = util.AverageMeter()
    NSP_nll_meter = util.AverageMeter()

    model.eval()
    correct_preds_MLM = 0
    correct_preds_NSP = 0
    total_preds_MLM = 0
    total_preds_NSP = 0
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, is_next, mask_1, masked_words_1, mask_2, masked_words_2 in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            is_next = is_next.to(device)
            mask_1 = tuple(x.to(device) for x in mask_1)
            mask_2 = tuple(x.to(device) for x in mask_2)
            masked_words_1 = masked_words_1.to(device)
            masked_words_2 = masked_words_2.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            MLM_logits_1, MLM_logits_2, NSP_logits = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs, mask_1, mask_2)
            MLM_logits = torch.cat((MLM_logits_1, MLM_logits_2), dim=0)
            masked_words = torch.cat((masked_words_1, masked_words_2), dim=0)

            MLM_loss = F.cross_entropy(input=MLM_logits, target=masked_words)
            NSP_loss = args.NSP_weight * F.binary_cross_entropy_with_logits(input=NSP_logits, target=is_next.float())
            loss = MLM_loss + NSP_loss
            nll_meter.update(loss.item(), batch_size)
            MLM_nll_meter.update(MLM_loss.item(), batch_size)
            NSP_nll_meter.update(NSP_loss.item(), batch_size)

            MLM_preds = torch.argmax(MLM_logits, dim=1)
            NSP_preds = (NSP_logits > 0.5)
            correct_preds_MLM += torch.sum(MLM_preds == masked_words).item()
            total_preds_MLM += MLM_preds.shape[0]
            correct_preds_NSP += torch.sum(NSP_preds == is_next.byte()).item()
            total_preds_NSP += NSP_preds.shape[0]

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(MLM_NLL=MLM_loss.item(),
                                     NSP_NLL=NSP_loss.item(),
                                     NLL=loss.item())

    model.train()

    results_list = [('NLL', nll_meter.avg),
                    ('MLM_NLL', MLM_nll_meter.avg),
                    ('NSP_NLL', NSP_nll_meter.avg),
                    ('Accuracy MLM', correct_preds_MLM / total_preds_MLM),
                    ('Accuracy NSP', correct_preds_NSP / total_preds_NSP),
                    ('Accuracy', (correct_preds_MLM / total_preds_MLM + correct_preds_NSP / total_preds_NSP) / 2)]

    results = OrderedDict(results_list)

    return results


if __name__ == '__main__':
    main(get_train_args())

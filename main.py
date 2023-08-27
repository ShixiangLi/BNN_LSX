import json
import logging
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable

import models_cifar
import models_imagenet
from utils import *


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if config['gpus'] is not None:
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        input_var = Variable(inputs.type(config['type']))
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['print_freq'] == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(data_loader),
                phase='TRAINING' if training else 'EVALUATING',
                batch_time=batch_time,
                data_time=data_time, loss=losses,
                top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    model.train()
    return forward(data_loader, model, criterion, epoch, training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    model.eval()
    return forward(data_loader, model, criterion, epoch, training=False, optimizer=None)


# --------------------------------------------读取训练配置文件-------------------------------------------------
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# --------------------------------------------配置log路径-------------------------------------------------
save_path = os.path.join(config['results_dir'], str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(os.path.join(save_path, 'config.txt'), 'w') as args_file:
    args_file.write(str(datetime.datetime.now()) + '\n\n')
    for args_n, args_v in config.items():
        args_v = '' if not args_v and not isinstance(args_v, int) else args_v
        args_file.write(str(args_n) + ':  ' + str(args_v) + '\n')

setup_logging(os.path.join(save_path, 'logger.log'), filemode='a')

# --------------------------------------------准备数据集-------------------------------------------------------
train_loader, val_loader = dataset.load_data(
    dataset=config['dataset'],
    data_path=config['data_path'],
    batch_size=config['batch_size'],
    batch_size_test=config['batch_size_test'],
    num_workers=config['workers'],
)

# --------------------------------------------准备模型-------------------------------------------------------
if config['dataset'] == 'tinyimagenet':
    num_classes = 200
    model_zoo = 'models_imagenet.'
elif config['dataset'] == 'imagenet':
    num_classes = 1000
    model_zoo = 'models_imagenet.'
elif config['dataset'] == 'cifar10':
    num_classes = 10
    model_zoo = 'models_cifar.'
elif config['dataset'] == 'cifar100':
    num_classes = 100
    model_zoo = 'models_cifar.'
else:
    logging.error('Dataset %s is not supported.' % config['dataset'])
    raise ValueError('Dataset %s is not supported.' % config['dataset'])

model = eval(model_zoo + config['model'])(num_classes=num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model = model.type(config['type'])

# --------------------------------------------准备优化器-------------------------------------------------------
if config['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': config['lr']}],
                                lr=config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay'])
elif config['optimizer'] == 'adam':
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': config['lr']}],
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])
else:
    logging.error("Optimizer '%s' not defined.", config['optimizer'])
    raise ValueError('Optimizer %s is not supported.' % config['optimizer'])

# --------------------------------------------准备学习率调整器-------------------------------------------------------
if config['lr_type'] == 'cos':
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'] - config['warm_up'] * 4,
                                                              eta_min=0,
                                                              last_epoch=config['start_epoch'])
elif config['lr_type'] == 'step':
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config['lr_decay_step'], gamma=0.1, last_epoch=-1)
elif config['lr_type'] == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (
            1.0 - (epoch - config['warm_up'] * 4) / (config['epochs'] - config['warm_up'] * 4)), last_epoch=-1)
else:
    logging.error("lr_type '%s' not defined.", config['lr_type'])
    raise ValueError('lr_type %s is not supported.' % config['lr_type'])

# --------------------------------------------准备损失函数-------------------------------------------------------
criterion = nn.CrossEntropyLoss().cuda()
criterion = criterion.type(config['type'])

# --------------------------------------------开始训练-------------------------------------------------------
best_prec1 = 0
for epoch in range(config['start_epoch'] + 1, config['epochs']):
    time_start = datetime.datetime.now()
    # warm up
    if config['warm_up'] and epoch < 5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['lr'] * (epoch + 1) / 5
    for param_group in optimizer.param_groups:
        logging.info('lr: %s', param_group['lr'])
        break

    # training
    train_loss, train_prec1, train_prec5 = train(train_loader, model, criterion, epoch, optimizer)

    # adjust Lr
    if epoch >= 4 * config['warm_up']:
        lr_scheduler.step()

    # evaluating
    with torch.no_grad():
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, epoch)

    # remember best prec
    is_best = val_prec1 > best_prec1
    if is_best:
        best_prec1 = max(val_prec1, best_prec1)
        best_epoch = epoch
        best_loss = val_loss

    # save model
    if epoch % 1 == 0:
        model_state_dict = model.module.state_dict() if len(config['gpus']) > 1 else model.state_dict()
        model_optimizer = optimizer.state_dict()
        model_scheduler = lr_scheduler.state_dict()
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config['model'],
            'state_dict': model_state_dict,
            'best_prec1': best_prec1,
            'optimizer': model_optimizer,
            'lr_scheduler': model_scheduler,
        }, is_best, path=save_path)

    if config['time_estimate'] > 0 and epoch % config['time_estimate'] == 0:
        time_end = datetime.datetime.now()
        cost_time, finish_time = get_time(time_end - time_start, epoch, config['epochs'])
        logging.info('Time cost: ' + cost_time + '\t'
                                                 'Time of Finish: ' + finish_time)

    logging.info('\n Epoch: {0}\t'
                 'Training Loss {train_loss:.4f} \t'
                 'Training Prec@1 {train_prec1:.3f} \t'
                 'Training Prec@5 {train_prec5:.3f} \t'
                 'Validation Loss {val_loss:.4f} \t'
                 'Validation Prec@1 {val_prec1:.3f} \t'
                 'Validation Prec@5 {val_prec5:.3f} \n'
                 .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                         train_prec1=train_prec1, val_prec1=val_prec1,
                         train_prec5=train_prec5, val_prec5=val_prec5))

    logging.info('*' * 50 + 'DONE' + '*' * 50)
    logging.info('\n Best_Epoch: {0}\t'
                 'Best_Prec1 {prec1:.4f} \t'
                 'Best_Loss {loss:.3f} \t'
                 .format(best_epoch + 1, prec1=best_prec1, loss=best_loss))

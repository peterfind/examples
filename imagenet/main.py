import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from PIL import ImageFile
import numpy as np
import time
from tensorboardX import SummaryWriter

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')  # original resnet18
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')  # original 256
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')  # original 0.1
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='0', type=int,
                    help='GPU id to use.')
parser.add_argument('--checkpoint_dir', default='./checkpoint_test')
parser.add_argument('--class_type', default='Style')
parser.add_argument('--tensorboard_print_freq', default=100)
parser.add_argument('--tensorboard_log_dir', default='./runs')
parser.add_argument('--dispaly_images', action='store_true')

best_prec1 = 0

num_class = {'Genre': 10,
             'Artist': 23,
             'Style': 27,
             'Style18': 18}
csv_path = {'Genre_train_csv':'/media/gisdom/2TB_2/luyue/ArtGAN/WikiArt Dataset/Genre/genre_train.csv',
            'Genre_validate_csv': '/media/gisdom/2TB_2/luyue/ArtGAN/WikiArt Dataset/Genre/genre_train.csv',
            'Artist_train_csv': '/media/gisdom/2TB_2/luyue/ArtGAN/WikiArt Dataset/Artist/artist_train.csv',
            'Artist_validate_csv': '/media/gisdom/2TB_2/luyue/ArtGAN/WikiArt Dataset/Artist/artist_val.csv',
            'Style_train_csv': '/media/gisdom/2TB_2/luyue/ArtGAN/WikiArt Dataset/Style/style_train.csv',
            'Style_validate_csv': '/media/gisdom/2TB_2/luyue/ArtGAN/WikiArt Dataset/Style/style_val.csv',
            'Style18_train_csv': '/media/gisdom/2TB_2/luyue/ArtGAN/WikiArt Dataset/Style/style_train_18.csv',
            'Style18_validate_csv': '/media/gisdom/2TB_2/luyue/ArtGAN/WikiArt Dataset/Style/style_val_18.csv'
            }

class WikiArtDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_list = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list.iloc[idx, 0])
        image = Image.open(img_name)
        tmp_transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor()])
        ori_image = tmp_transform(image)
        label = self.image_list.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label,
                  'ori_image': ori_image,
                  'img_name': self.image_list.iloc[idx, 0]}
        return sample


def main():
    wikiart_img_root = '/media/gisdom/2TB_2/luyue/datasets/wikiart'
    global args, best_prec1, writer
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    for name, value in vars(args).items():
        print('{} = {}'.format(name, value))

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    model.classifier._modules['6'] = nn.Linear(4096, num_class[args.class_type])
    state = model.state_dict()
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.pretrained and args.arch == 'vgg19':
        features_all = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        features_run = [10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        features_run = [str(value) for value in features_run]
        for param in model.parameters():
            param.requires_grad = False
        gens = [model.features._modules[name].parameters() for name in features_run]
        gens.append(model.classifier.parameters())
        for gen in gens:
            for param in gen:
                param.requires_grad = True
        # dict_params = [{'params': gen} for gen in gens]
        dict_params = [{'params': model.features._modules[name].parameters()} for name in features_run]
        dict_params.append({'params':model.classifier.parameters()})
        print('Fix some layer in pretrained model.')
    else:
        dict_params = model.parameters()
    optimizer = torch.optim.SGD(dict_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])
    val_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])
    train_dataset = WikiArtDataset(csv_file=csv_path[args.class_type+'_train_csv'], root_dir=wikiart_img_root,
                                   transform=train_transform)
    validate_trainSet = WikiArtDataset(csv_file=csv_path[args.class_type + '_train_csv'], root_dir=wikiart_img_root,
                                      transform=val_transform)
    validate_dataset = WikiArtDataset(csv_file=csv_path[args.class_type+'_validate_csv'], root_dir=wikiart_img_root,
                                      transform=val_transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(validate_dataset,
                                             batch_size=64, shuffle=False,
                                             num_workers=8, pin_memory=True)
    val_trainset_loader = torch.utils.data.DataLoader(validate_trainSet,
                                             batch_size=64, shuffle=False,
                                             num_workers=8, pin_memory=True)

    if args.evaluate:
        validate('validationSet', val_loader, model, criterion)
        validate('trainSet', val_trainset_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        train_prec1 = validate('trainSet', val_trainset_loader, model, criterion, epoch)

        # evaluate on validation set
        prec1 = validate('validationSet', val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Saving models, Donot interrupt...')
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            filename=os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar'))
        if epoch % 10 == 0:
            shutil.copyfile(os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar'),
                            os.path.join(args.checkpoint_dir, 'checkpoint_epoch{}.pth.tar'.format(epoch)))
        print('Saving models completed.')


def train(train_loader, model, criterion, optimizer, epoch):
    print(time.asctime(time.localtime(time.time())))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss1 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    param_watcher = ParameterWatcher(model.state_dict())
    for i, data in enumerate(train_loader):
        writer_idx = i + epoch*len(train_loader)
        input = data['image']
        target = data['label']
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        loss1.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i%args.tensorboard_print_freq==0:
            writer.add_scalar('realtime_training/loss.val', loss1.val, writer_idx)
            writer.add_scalar('realtime_training/loss.avg', loss1.avg, writer_idx)
            writer.add_scalar('realtime_training/top1.avg', top1.avg, writer_idx)
            writer.add_scalar('realtime_training/top5.avg', top5.avg, writer_idx)

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Prec@1 ({top1:.3f})\t'
                  'Prec@5 ({top5:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss1=loss1, top1=top1.avg, top5=top5.avg))
        if i % 100 == 0 and args.dispaly_images:
            writer.add_image('epoch{:0>3d}/iter{:0>5d}_0oriImage'.format(epoch, i), data['ori_image'])
            writer.add_image('epoch{:0>3d}/iter{:0>5d}_1image'.format(epoch, i), data['image'])
    param_watcher.update(model.state_dict(), epoch)
    print(time.asctime(time.localtime(time.time())))


def validate(name, val_loader, model, criterion, epoch=0):
    print(time.asctime(time.localtime(time.time())))
    print('Validate on {}'.format(name))
    batch_time = AverageMeter()
    # losses = AverageGroup(size=num_class[args.class_type])
    top1s = AverageGroup(size=num_class[args.class_type])
    top5s = AverageGroup(size=num_class[args.class_type])
    losses = AverageGroup(size=num_class[args.class_type])

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            input = data['image']
            target = data['label']
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            losses_data = [criterion(output[i].view(1,-1), target[i].view(-1)) for i in range(len(output))]
            losses.update(losses_data, target)

            # measure accuracy and record loss
            prec1s, prec5s = accuracyes(output, target, topk=(1, 5))
            top1s.update(prec1s, target)
            top5s.update(prec5s, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader) - 1:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss ({losses.avg:.4f})\t'
                      'Prec@1 ({top1s.avg:.3f})\t'
                      'Prec@5 ({top5s.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, losses=losses,
                    top1s=top1s, top5s=top5s))

        writer.add_scalar('{}/losses.avg'.format(name), losses.avg, epoch)
        writer.add_scalar('{}/top1s.avg'.format(name), top1s.avg, epoch)
        writer.add_scalar('{}/top5s.avg'.format(name), top5s.avg, epoch)
        for idx, value in enumerate(top1s.avgs):
            writer.add_scalar('{}_class_top1s/{}'.format(name, idx), value, epoch)
        for idx, value in enumerate(top5s.avgs):
            writer.add_scalar('{}_class_top5s/{}'.format(name, idx), value, epoch)
        for idx, value in enumerate(losses.avgs):
            writer.add_scalar('{}_class_losses/{}'.format(name, idx), value, epoch)
        for idx, value in enumerate(top1s.counts):
            writer.add_scalar('{}_class_counts/{}'.format(name, idx), value, epoch)
        writer.add_scalar('{}_class_counts/all'.format(name), top1s.count, epoch)
    print(time.asctime(time.localtime(time.time())))
    return top1s.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.checkpoint_dir, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum1 = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum1 += val * n
        self.count += n
        self.avg = self.sum1 / self.count


class AverageGroup(object):
    def __init__(self, size=1):
        self.reset(size)

    def reset(self, size=1):
        self.avgs = np.zeros(size)
        self.sums = np.zeros(size)
        self.counts = np.zeros(size)
        self.count = 0
        self.sum1 = 0
        self.avg = 0

    def update(self, val, idx):
        self.sum1 += sum(val)
        self.count += len(idx)
        self.avg = self.sum1/self.count
        for i in range(len(idx)):
            self.sums[idx[i]] += val[i]
            self.counts[idx[i]] += 1
            self.avgs[idx[i]] = self.sums[idx[i]] / self.counts[idx[i]]

class ParameterWatcher(object):
    def __init__(self, state_dict):
        self.state_dict = {}
        for k in state_dict.keys():
            self.state_dict[k] = state_dict[k].cpu()
        self.state_dict_last = self.state_dict.copy()

    def update(self, state_dict, writer_idx):
        self.state_dict_last = self.state_dict.copy()
        for k in state_dict.keys():
            self.state_dict[k] = state_dict[k].cpu()
        # print('Parameter changes:')
        # for key in self.state_dict.keys():
        #     # count = 1
        #     # tmp_size = self.state_dict[key].shape
        #     # for x in tmp_size:
        #     #     count = count*x
        #     print('{}:{:.3f}\t'.format(key,torch.norm(self.state_dict[key]-self.state_dict_last[key], p=1)), end='')
        # print(' ')
        for key in self.state_dict.keys():
            writer.add_scalar('param.1norm/'+key,
                              torch.norm(self.state_dict[key]-self.state_dict_last[key],
                                         p=1), writer_idx)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracyes(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        # for k in topk:
        #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        # return res

        return [correct[:k].float().sum(0)*100.0 for k in topk]



if __name__ == '__main__':
    main()

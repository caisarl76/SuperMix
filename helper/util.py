from __future__ import print_function

import torch
import numpy as np
import datetime, os


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


class Logger:
    def __init__(self, dir, var_names=None, format=None, args=None  ):
        self.dir = dir
        self.var_names = var_names
        self.format = format
        self.vars = []

        # create the log folder
        if not os.path.exists(dir):
            os.makedirs(dir)

        file = open(dir + '/log.txt', 'w')
        file.write('Log file created on ' + str(datetime.datetime.now()) + '\n\n')

        dict = {}
        for arg in vars(args):
            dict[arg] = str(getattr(args, arg))

        for d in sorted(dict.keys()):
            file.write(d+ ' : ' + dict[d] + '\n')
        file.write('\n')
        file.close()

    def store(self, vars, log=False):
        self.vars = self.vars + vars
        if log:
            self.log()

    def log(self):

        vars = self.vars
        file = open(self.dir + '/log.txt', 'a')
        st = ''
        for i in range(len(vars)):
            st += self.var_names[i] + ': ' + self.format[i] % (vars[i]) + ', '
        st += 'time: ' + str(datetime.datetime.now()) + '\n'
        file.write(st)
        file.close()
        self.vars = []


if __name__ == '__main__':

    pass

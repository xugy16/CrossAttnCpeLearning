"""
To record the running process.
"""


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, format=':f'):
        self.name = name
        self.format = format
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

    def __str__(self):
        fmtstr = '{name} {val' + self.format + '} ({avg' + self.format + '})'
        return fmtstr.format(**self.__dict__)
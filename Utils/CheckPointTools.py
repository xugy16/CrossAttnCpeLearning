import os
import torch


def load_current_best_ckpt(cpkt_file, net):
    """
    Two cases:
    1. exist, then load the parameters.
    2. not exist, just return the net.
    Not loading the best model
    """
    cur_best_close_accu = 0.0
    if os.path.exists(cpkt_file):
        checkpoint = torch.load(cpkt_file)
        # Current  best model
        cur_best_close_accu = checkpoint['best_measure']
        print("current best accu is {}".format(cur_best_close_accu))
        # net.load_state_dict(checkpoint['state_dict'], strict=False)
    return net,  cur_best_close_accu



def save_current_best_ckpt(config, model, optimizer, epoch, best_measure, fname):
    """
    Basic informaiton we need.
    """
    torch.save( {'config': config,
                 'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'best_measure': best_measure,
                 'optimizer': optimizer.state_dict()},  fname)


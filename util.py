import json
import os
import shutil
import sys


def mkdir_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def yes_no_input():
    while True:
        choice = raw_input("Please respond with 'yes' or 'no' [y/N]: ").lower()
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False


def check_if_done(filename):
    if os.path.exists(filename):
        print ("%s already exists. Is it O.K. to overwrite it and start this program?" % filename)
        if not yes_no_input():
            raise Exception("Please restart training after you set args.savename differently!")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    import torch

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def exec_eval(tgt_dataset, label_outdir):
    print ("-" * 10 + " Evaluation using this command " + "-" * 10)
    eval_str = "python eval.py %s %s" % (tgt_dataset, label_outdir)
    print (eval_str)
    import subprocess
    subprocess.call(eval_str, shell=True)


def calc_entropy(output):
    import torch
    import torch.nn.functional as F
    output = F.softmax(output)
    return -torch.mean(output * torch.log(output + 1e-6))


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


def save_dic_to_json(dic, fn, verbose=True):
    dic = {str(k): v for k, v in dic.items()}

    with open(fn, "w") as f:
        json_str = json.dumps(dic, sort_keys=True, indent=4)
        if verbose:
            print (json_str)
        f.write(json_str)
    print ("param file '%s' was saved!" % fn)


def emphasize_str(string):
    print ('#' * 100)
    print (string)
    print ('#' * 100)


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch >= num_epochs * 0.75:
        lr *= decay_rate ** 2
    elif epoch >= num_epochs * 0.5:
        lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_class_weight_from_file(n_class, weight_filename=None, add_bg_loss=False):
    import torch
    weight = torch.ones(n_class)
    if weight_filename:
        import pandas as pd

        loss_df = pd.read_csv(weight_filename)
        loss_df.sort_values("class_id", inplace=True)
        weight *= torch.FloatTensor(loss_df.weight.values)

    if not add_bg_loss:
        weight[n_class - 1] = 0  # Ignore background loss
    return weight


def save_colorized_lbl(idxed_lbl, outfn, dataset="city"):
    """
    Colorize and Save label data (1ch img contains 0~N_class+ 255(Void))
    :param idxed_lbl:
    :param outfn:
    :param dataset:
    :return:
    """
    import numpy as np

    #  Save visualized predicted pixel labels(pngs)
    if dataset in ["city16", "synthia"]:
        info_json_fn = "./dataset/synthia2cityscapes_info.json"
    elif dataset == "2d3d":
        info_json_fn = "./dataset/2d3d_info.json"
    elif dataset in ["nyu", "suncg"]:
        info_json_fn = "./dataset/nyu_info.json"
    else:
        info_json_fn = "./dataset/city_info.json"

    # Save visualized predicted pixel labels(pngs)
    with open(info_json_fn) as f:
        city_info_dic = json.load(f)

    palette = np.array(city_info_dic['palette'], dtype=np.uint8)
    idxed_lbl.putpalette(palette.flatten())
    idxed_lbl.save(outfn)


def set_debugger_org():
    if not sys.excepthook == sys.__excepthook__:
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(call_pdb=True)


def set_debugger_org_frc():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)


def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

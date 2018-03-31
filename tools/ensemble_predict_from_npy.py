"""Ensemble predict."""

import argparse
import os

import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from transform import Colorize
from util import mkdir_if_not_exist

N_CLASS = 19


def ensemble_predict(prob_fns, outfile='sample.png',
                     out_npy_file='sample.npy', out_vis_file='sample_vis.png',
                     method='averaging', out_shape=(2048, 1024)):
    """Output predict file from two npy files by the given method."""
    probs = [np.load(prob_fn) for prob_fn in prob_fns]

    if method == 'averaging':
        prob = sum(probs) / len(probs)
    elif method == 'nms':
        prob = np.max(probs, 0)

    # -- output npy --
    np.save(out_npy_file, prob)

    # -- output label-predict --
    pred = prob[:N_CLASS].argmax(0)
    img = Image.fromarray(np.uint8(pred))
    img = img.resize(out_shape, Image.NEAREST)
    img.save(outfile)

    # -- output vis-predict --

    # ToTensor function: ndarray -> Tensor
    #   * H, W, C -> C, H, W
    #   * 0, 255 -> 0, 1
    # prob.shape == (20, 512, 1024)
    prob_for_tensor = np.transpose(prob[:N_CLASS], (1, 2, 0))
    prob_tensor = ToTensor()(prob_for_tensor)

    pred = prob_tensor.max(0)[1]
    pred = pred.view(1, pred.size()[0], pred.size()[1])

    # Colorize function: Tensor -> Tensor
    vis_tensor = Colorize()(pred)

    # Tensor -> ndarray -> image(save)
    vis = np.transpose(vis_tensor.numpy(), (1, 2, 0))
    vis_img = Image.fromarray(vis, 'RGB')
    vis_img = vis_img.resize(out_shape, Image.NEAREST)
    vis_img.save(out_vis_file)


def main(npydirs,
         out_directory='ensemble_results',
         method='averaging', mode='test'):
    """
    Ensemble.

    1. get all npy files from given directory.
    2. ensemble each file and output predict png file.
    """

    out_shape = (2048, 1024) if mode == 'valid' else (1280, 720)

    out_label_dir = os.path.join(out_directory, 'label')
    out_vis_dir = os.path.join(out_directory, 'vis')
    out_prob_dir = os.path.join(out_directory, 'prob')

    mkdir_if_not_exist(out_label_dir)
    mkdir_if_not_exist(out_vis_dir)
    mkdir_if_not_exist(out_prob_dir)

    print('- npy_directory_list')
    print(npydirs)

    print('- method')
    print(method)

    print('- mode')
    print(mode, out_shape)

    prob_filenames = os.listdir(npydirs[0])

    print(len(prob_filenames))
    print('Ensembling ...')
    for i, prob_filename in tqdm(enumerate(prob_filenames)):
        png_filename = prob_filename.replace('npy', 'png')

        prob_fns = [os.path.join(npydir, prob_filename) for npydir in npydirs]

        ensemble_predict(prob_fns,
                         os.path.join(out_label_dir, png_filename),
                         os.path.join(out_prob_dir, prob_filename),
                         os.path.join(out_vis_dir, png_filename),
                         method, out_shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npydirs', type=str, nargs='+',
                        help='result directory that contains probability npys')
    parser.add_argument('--method', default='averaging',
                        help='"averaging" or "nms"', choices=["averaging", "nms"])
    parser.add_argument('-d', '--out_directory', default='ensemble_results')
    parser.add_argument('--mode', default='test', choices=["test", "valid"])

    args = parser.parse_args()

    assert len(args.npydirs) > 1

    main(args.npydirs, args.out_directory, args.method, args.mode)

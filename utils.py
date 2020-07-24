'''
Minor utilities for testing.
You do not have to import this code to use core.py
'''

import time
import numpy as np
import imageio

import torch


class Timer(object):

    def __init__(self, msg: str) -> None:
        self.msg = msg.replace('{}', '{:.6f}s')
        self.tic = None
        return

    def __enter__(self) -> None:
        self.tic = time.time()
        return

    def __exit__(self, *args, **kwargs) -> None:
        toc = time.time() - self.tic
        print('\n' + self.msg.format(toc))
        return


def np2tensor(x: np.array) -> torch.Tensor:
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x)
    with torch.no_grad():
        while x.dim() < 4:
            x.unsqueeze_(0)

        x = x.float() / 255

    return x

def tensor2np(x: torch.Tensor) -> np.array:
    with torch.no_grad():
        x = 255 * x
        x = x.round().clamp(min=0, max=255).byte()
        x = x.squeeze(0)

    x = x.cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    x = np.ascontiguousarray(x)
    return x

def get_img(img_path: str) -> torch.Tensor:
    x = imageio.imread(img_path)
    x = np2tensor(x)
    return x

def save_img(x: torch.Tensor, img_path: str) -> None:
    x = tensor2np(x)
    imageio.imwrite(img_path, x)
    return

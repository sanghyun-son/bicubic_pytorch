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


def get_img(img_path: str) -> torch.Tensor:
    img = imageio.imread(img_path)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    while img.dim() < 4:
        img.unsqueeze_(0)

    img = img.float() / 255
    return img

def save_img(x: torch.Tensor, img_path: str) -> None:
    with torch.no_grad():
        x = 255 * x
        x = x.round().clamp(min=0, max=255).byte()
        x = x.squeeze(0)

    x = x.cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    imageio.imwrite(img_path, x)
    return

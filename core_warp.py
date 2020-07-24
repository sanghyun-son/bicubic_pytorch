import math
import typing

import core

import cv2

import torch
from torch.nn import functional as F


def warp_by_size(
        x: torch.Tensor,
        m: torch.Tensor,
        sizes: typing.Tuple[int, int],
        kernel: typing.Union[str, torch.Tensor]='cubic',
        padding_type: str='reflect',
        fill_value: int=0) -> torch.Tensor:

    h_orig = x.size(-2)
    w_orig = x.size(-1)
    h, w = sizes
    dkwargs = {'dtype': x.dtype, 'device': x.device}
    # Construct the target coordinate
    pos_i = torch.linspace(start=0, end=(h - 1), steps=h, **dkwargs)
    pos_j = torch.linspace(start=0, end=(w - 1), steps=w, **dkwargs)
    ones = x.new_ones(h * w)
    with torch.no_grad():
        pos_i = pos_i.view(-1, 1)
        pos_i = pos_i.repeat(1, w)
        pos_j = pos_j.view(1, -1)
        pos_j = pos_j.repeat(h, 1)
        pos = torch.stack([pos_j.view(-1), pos_i.view(-1), ones], dim=0)
        m_inv = m.inverse()
        # Map to the source coordinate
        pos_backward = torch.matmul(m_inv, pos)
        pos_backward = pos_backward[:2] / pos_backward[-1, :]
        # Out of the image
        pos_under = (pos_backward < -0.5).any(dim=0)
        pos_over_h = (pos_backward[1] >= h_orig - 0.5)
        pos_over_w = (pos_backward[0] >= w_orig - 0.5)
        pos_out = (pos_under + pos_over_h + pos_over_w).float()

        if isinstance(kernel, str):
            if kernel == 'nearest':
                kernel_size = 1
                pos_backward = pos_backward.round()
                idx = pos_backward[0] + x.size(-1) * pos_backward[1]
            elif kernel == 'bilinear':
                kernel_size = 2
            elif kernel == 'cubic':
                kernel_size = 4
        else:
            pass

        idx = -1 * pos_out + (1 - pos_out) * idx
        idx = idx.long()
        idx = idx.clamp(min=-1)

    x = F.unfold(x, (kernel_size, kernel_size))
    fill_value = x.new_full((x.size(0), x.size(1), 1), fill_value=fill_value)
    x = torch.cat((x, fill_value), dim=-1)
    # (B, kernel_size^2, H x W)
    if kernel == 'nearest':
        x = x[..., idx]

    x = x.view(-1, 1, h, w)
    return x

def warp(
        x: torch.Tensor,
        m: torch.Tensor,
        sizes: typing.Union[typing.Tuple[int, int], str, None]=None,
        kernel: typing.Union[str, torch.Tensor]='cubic',
        padding_type: str='reflect',
        fill_value: int=0) -> torch.Tensor:

    x, b, c, h, w = core.reshape_input(x)
    x, dtype = core.cast_input(x)
    m = m.to(x.device)

    if sizes is None:
        sizes = (h, w)
    elif isinstance(sizes, str) and sizes == 'auto':
        with torch.no_grad():
            corners = m.new_tensor([
                [-0.5, -0.5, w - 0.5, w - 0.5],
                [-0.5, h - 0.5, -0.5, h - 0.5],
                [1, 1, 1, 1],
            ])
            corners = torch.matmul(m, corners)
            corners = corners / corners[-1, :]
            y_min = corners[1].min() + 0.5
            x_min = corners[0].min() + 0.5
            h_new = math.floor(corners[1].max() - y_min + 0.5)
            w_new = math.floor(corners[0].max() - x_min + 0.5)
            m_comp = m.new_tensor([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
            m = torch.matmul(m_comp, m)
            sizes = (h_new, w_new)

    elif not isinstance(sizes, tuple):
        raise ValueError('sizes:', sizes, 'is not supported!')

    x = warp_by_size(
        x,
        m,
        sizes,
        kernel=kernel,
        padding_type=padding_type,
        fill_value=fill_value,
    )
    x = core.reshape_output(x, b, c)
    x = core.cast_output(x, dtype)
    return x


if __name__ == '__main__':
    import os
    import utils
    #x = torch.arange(64).float().view(1, 1, 8, 8)
    x = utils.get_img('example/butterfly.png')
    #m = torch.Tensor([[3.2, 0.016, -68], [1.23, 1.7, -54], [0.008, 0.0001, 1]])
    m = torch.Tensor([[2.33e-01, 3.97e-3, 3], [-4.49e-1, 2.49e-1, 1.15e2], [-2.95e-3, 1.55e-5, 1]])
    #m = torch.Tensor([[2, 0, 1], [0, 2, 0], [0, 0, 1]])
    y = warp(x, m, sizes='auto', kernel='nearest', fill_value=0)
    #y = warp(x, m, kernel='nearest', fill_value=0)
    os.makedirs('dummy', exist_ok=True)
    utils.save_img(y, 'dummy/warp.png')


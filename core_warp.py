import math
import typing

import core

import cv2

import torch
from torch.nn import functional as F

def contribution_2d(x: torch.Tensor, kernel: str='cubic') -> torch.Tensor:
    '''
    Args:
        x (torch.Tensor): (2, k, N), where x[0] is the x-coordinate.
        kernel (str):

    Return
        torch.Tensor: (k^2, N)
    '''
    if kernel == 'nearest':
        weight = core.nearest_contribution(x)
    elif kernel == 'bilinear':
        weight = core.linear_contribution(x)
    elif kernel == 'bicubic':
        weight = core.cubic_contribution(x)

    weight_x = weight[0].unsqueeze(0)
    weight_y = weight[1].unsqueeze(1)
    weight = weight_x * weight_y
    weight = weight.view(-1, weight.size(-1))
    weight = weight / weight.sum(0, keepdim=True)
    return weight

def warp_by_size(
        x: torch.Tensor,
        m: torch.Tensor,
        sizes: typing.Tuple[int, int],
        kernel: typing.Union[str, torch.Tensor]='bicubic',
        padding_type: str='reflect',
        fill_value: int=0) -> torch.Tensor:

    h_orig = x.size(-2)
    w_orig = x.size(-1)
    h, w = sizes
    dkwargs = {'dtype': x.dtype, 'device': x.device, 'requires_grad': False}
    # Construct the target coordinates
    # The target coordinates do not require gradients
    pos_i = torch.arange(h, **dkwargs)
    pos_j = torch.arange(w, **dkwargs)
    pos_i = pos_i.view(-1, 1).repeat(1, w).view(-1)
    pos_j = pos_j.view(1, -1).repeat(h, 1).view(-1)
    # Map the target coordinates to the source coordinates
    # This implements the backward warping
    pos = torch.stack([pos_j, pos_i, torch.ones_like(pos_i)], dim=0)
    pos_bw = torch.matmul(m.inverse(), pos)
    pos_bw = pos_bw[:2] / pos_bw[-1, :]
    # Out of the image
    pos_over = pos_bw.new_tensor([w_orig, h_orig]).unsqueeze(-1)
    pos_out = torch.logical_or(pos_bw.lt(-0.5), pos_bw.ge(pos_over - 0.5))
    pos_out = pos_out.any(0).float()

    kernels = {'nearest': 1, 'bilinear': 2, 'bicubic': 4}
    if isinstance(kernel, str):
        if kernel in kernels:
            kernel_size = kernels[kernel]
        else:
            raise ValueError('kernel: {} is not supported!'.format(kernel))

        if kernel_size % 2 == 0:
            pos_discrete = pos_bw.ceil()
            pos_frac = pos_bw - pos_bw.floor()
        else:
            pos_discrete = pos_bw.round()
            pos_frac = torch.ones_like(pos_discrete)

        pad = kernel_size // 2
        # (2, 1, HW)
        pos_frac.unsqueeze_(1)
        # (2, k, 1)
        pos_w = torch.linspace(-pad + 1, pad, kernel_size, **dkwargs)
        pos_w = pos_frac - pos_w.view(1, -1, 1).repeat(2, 1, 1)
        # (1, k^2, HW)
        weight = contribution_2d(pos_w, kernel=kernel)
        weight.unsqueeze_(0)
    else:
        pass

    # Calculate the exact sampling point
    if kernel == 'nearest':
        idx = pos_discrete[0] + x.size(-1) * pos_discrete[1]
    else:
        idx = pos_discrete[0] + (x.size(-1) + 1) * pos_discrete[1]

    # Remove the outside region
    idx = -1 * pos_out + (1 - pos_out) * idx
    idx = idx.long()
    idx = idx.clamp(min=-1)

    x = core.padding(x, -2, pad, pad, padding_type=padding_type)
    x = core.padding(x, -1, pad, pad, padding_type=padding_type)
    x = F.unfold(x, (kernel_size, kernel_size))
    '''
    for i in range(x.size(-1)):
        print(x[..., i].view(4, 4), weight[..., i].view(4, 4))
    '''
    fill_value = x.new_full((x.size(0), x.size(1), 1), fill_value=fill_value)
    x = torch.cat((x, fill_value), dim=-1)
    # (B, k^2, HW)
    x = x[..., idx]
    x = x * weight
    x = x.sum(dim=1, keepdim=True)
    x = x.view(-1, 1, h, w)
    return x

def warp(
        x: torch.Tensor,
        m: torch.Tensor,
        sizes: typing.Union[typing.Tuple[int, int], str, None]=None,
        kernel: typing.Union[str, torch.Tensor]='bicubic',
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
    #x = torch.arange(16).float().view(1, 1, 4, 4)
    x = utils.get_img('example/butterfly.png')
    m = torch.Tensor([[3.2, 0.016, -68], [1.23, 1.7, -54], [0.008, 0.0001, 1]])
    #m = torch.Tensor([[2.33e-01, 3.97e-3, 3], [-4.49e-1, 2.49e-1, 1.15e2], [-2.95e-3, 1.55e-5, 1]])
    m = torch.Tensor([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    y = warp(x, m, sizes='auto', kernel='bicubic', fill_value=0)
    #y = warp(x, m, kernel='nearest', fill_value=0)
    os.makedirs('dummy', exist_ok=True)
    #utils.save_img(y, 'dummy/warp.png')


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
        kernel: str='bicubic',
        padding_type: str='reflect',
        fill_value: int=0) -> torch.Tensor:

    kernels = {'nearest': 1, 'bilinear': 2, 'bicubic': 4}
    if kernel in kernels:
        k = kernels[kernel]
        pad = k // 2
    else:
        raise ValueError('kernel: {} is not supported!'.format(kernel))

    dkwargs = {'device': x.device, 'requires_grad': False}
    # Construct the target coordinates
    # The target coordinates do not require gradients
    pos = torch.arange(sizes[0] * sizes[1], **dkwargs)
    pos_i = (pos // sizes[1]).float()
    pos_j = (pos % sizes[1]).float()
    # Map the target coordinates to the source coordinates
    # This implements the backward warping
    pos_tar = torch.stack([pos_j, pos_i, torch.ones_like(pos_i)], dim=0)
    pos_src = torch.matmul(m.inverse(), pos_tar)
    pos_src = pos_src[:2] / pos_src[-1, :]
    # Out of the image
    pos_bound = pos_src.new_tensor([x.size(-1), x.size(-2)]) - 0.5
    pos_bound.unsqueeze_(-1)
    pos_in = torch.logical_and(pos_src.ge(-0.5), pos_src.lt(pos_bound))
    pos_in = pos_in.all(0)
    # Remove the outside region and compensate subpixel shift
    sub = (k % 2) / 2
    pos_src = pos_src[..., pos_in]
    pos_src_sub = pos_src - sub
    pos_discrete = pos_src_sub.ceil().long()
    pos_frac = pos_src_sub - pos_src.floor()
    pos_frac.unsqueeze_(1)
    # (2, 1, HW)
    pos_w = torch.linspace(pad - k + 1, pad, k, **dkwargs)
    pos_w = pos_w.view(1, -1, 1).repeat(2, 1, 1)
    pos_w = pos_frac - pos_w
    weight = contribution_2d(pos_w, kernel=kernel)
    weight.unsqueeze_(0)

    # Calculate the exact sampling point
    idx = pos_discrete[0] + (x.size(-1) + 1 - k % 2) * pos_discrete[1]

    # (B, k^2, HW)
    x = core.padding(x, -2, pad, pad, padding_type=padding_type)
    x = core.padding(x, -1, pad, pad, padding_type=padding_type)
    x = F.unfold(x, (k, k))
    sample = x[..., idx]

    y = sample * weight
    y = y.sum(dim=1)
    out = y.new_full((y.size(0), pos_in.size(0)), fill_value)
    out.masked_scatter_(pos_in, y)
    out = out.view(-1, 1, *sizes)
    return out

def warp(
        x: torch.Tensor,
        m: torch.Tensor,
        sizes: typing.Union[typing.Tuple[int, int], str, None]=None,
        kernel: str='bicubic',
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
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=16, linewidth=200)
    #x = torch.arange(64).float().view(1, 1, 8, 8)
    x = torch.arange(16).float().view(1, 1, 4, 4)
    #x = utils.get_img('example/butterfly.png')
    #x.requires_grad = True
    #m = torch.Tensor([[3.2, 0.016, -68], [1.23, 1.7, -54], [0.008, 0.0001, 1]])
    #m = torch.Tensor([[2.33e-01, 3.97e-3, 3], [-4.49e-1, 2.49e-1, 1.15e2], [-2.95e-3, 1.55e-5, 1]])
    m = torch.Tensor([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    y = warp(x, m, sizes='auto', kernel='bicubic', fill_value=-1)
    z = core.imresize(x, scale=2, kernel='cubic')
    print(y)
    print(z)
    #os.makedirs('dummy', exist_ok=True)
    #utils.save_img(y, 'dummy/warp.png')

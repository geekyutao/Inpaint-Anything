import torch
import math
import torch.nn.functional as F


def hann1d(sz: int, centered = True) -> torch.Tensor:
    """1D cosine window."""
    if centered:
        return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
    w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2)) * torch.arange(0, sz//2 + 1).float()))
    return torch.cat([w, w[1:sz-sz//2].flip((0,))])


def hann2d(sz: torch.Tensor, centered = True) -> torch.Tensor:
    """2D cosine window."""
    return hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * hann1d(sz[1].item(), centered).reshape(1, 1, 1, -1)


def hann2d_bias(sz: torch.Tensor, ctr_point: torch.Tensor, centered = True) -> torch.Tensor:
    """2D cosine window."""
    distance = torch.stack([ctr_point, sz-ctr_point], dim=0)
    max_distance, _ = distance.max(dim=0)

    hann1d_x = hann1d(max_distance[0].item() * 2, centered)
    hann1d_x = hann1d_x[max_distance[0] - distance[0, 0]: max_distance[0] + distance[1, 0]]
    hann1d_y = hann1d(max_distance[1].item() * 2, centered)
    hann1d_y = hann1d_y[max_distance[1] - distance[0, 1]: max_distance[1] + distance[1, 1]]

    return hann1d_y.reshape(1, 1, -1, 1) * hann1d_x.reshape(1, 1, 1, -1)



def hann2d_clipped(sz: torch.Tensor, effective_sz: torch.Tensor, centered = True) -> torch.Tensor:
    """1D clipped cosine window."""

    # Ensure that the difference is even
    effective_sz += (effective_sz - sz) % 2
    effective_window = hann1d(effective_sz[0].item(), True).reshape(1, 1, -1, 1) * hann1d(effective_sz[1].item(), True).reshape(1, 1, 1, -1)

    pad = (sz - effective_sz) // 2

    window = F.pad(effective_window, (pad[1].item(), pad[1].item(), pad[0].item(), pad[0].item()), 'replicate')

    if centered:
        return window
    else:
        mid = (sz / 2).int()
        window_shift_lr = torch.cat((window[:, :, :, mid[1]:], window[:, :, :, :mid[1]]), 3)
        return torch.cat((window_shift_lr[:, :, mid[0]:, :], window_shift_lr[:, :, :mid[0], :]), 2)


def gauss_fourier(sz: int, sigma: float, half: bool = False) -> torch.Tensor:
    if half:
        k = torch.arange(0, int(sz/2+1))
    else:
        k = torch.arange(-int((sz-1)/2), int(sz/2+1))
    return (math.sqrt(2*math.pi) * sigma / sz) * torch.exp(-2 * (math.pi * sigma * k.float() / sz)**2)


def gauss_spatial(sz, sigma, center=0, end_pad=0):
    k = torch.arange(-(sz-1)/2, (sz+1)/2+end_pad)
    return torch.exp(-1.0/(2*sigma**2) * (k - center)**2)


def label_function(sz: torch.Tensor, sigma: torch.Tensor):
    return gauss_fourier(sz[0].item(), sigma[0].item()).reshape(1, 1, -1, 1) * gauss_fourier(sz[1].item(), sigma[1].item(), True).reshape(1, 1, 1, -1)

def label_function_spatial(sz: torch.Tensor, sigma: torch.Tensor, center: torch.Tensor = torch.zeros(2), end_pad: torch.Tensor = torch.zeros(2)):
    """The origin is in the middle of the image."""
    return gauss_spatial(sz[0].item(), sigma[0].item(), center[0], end_pad[0].item()).reshape(1, 1, -1, 1) * \
           gauss_spatial(sz[1].item(), sigma[1].item(), center[1], end_pad[1].item()).reshape(1, 1, 1, -1)


def cubic_spline_fourier(f, a):
    """The continuous Fourier transform of a cubic spline kernel."""

    bf = (6*(1 - torch.cos(2 * math.pi * f)) + 3*a*(1 - torch.cos(4 * math.pi * f))
           - (6 + 8*a)*math.pi*f*torch.sin(2 * math.pi * f) - 2*a*math.pi*f*torch.sin(4 * math.pi * f)) \
         / (4 * math.pi**4 * f**4)

    bf[f == 0] = 1

    return bf

def max2d(a: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Computes maximum and argmax in the last two dimensions."""

    max_val_row, argmax_row = torch.max(a, dim=-2)
    max_val, argmax_col = torch.max(max_val_row, dim=-1)
    argmax_row = argmax_row.view(argmax_col.numel(),-1)[torch.arange(argmax_col.numel()), argmax_col.view(-1)]
    argmax_row = argmax_row.reshape(argmax_col.shape)
    argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)), -1)
    return max_val, argmax

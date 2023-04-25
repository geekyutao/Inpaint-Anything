import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


def generate_2d_relative_positional_encoding_index(z_shape, x_shape):
    '''
        z_shape: (z_h, z_w)
        x_shape: (x_h, x_w)
    '''
    z_2d_index_h, z_2d_index_w = torch.meshgrid(torch.arange(z_shape[0]), torch.arange(z_shape[1]))
    x_2d_index_h, x_2d_index_w = torch.meshgrid(torch.arange(x_shape[0]), torch.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    diff_h = z_2d_index_h[:, None] - x_2d_index_h[None, :]
    diff_w = z_2d_index_w[:, None] - x_2d_index_w[None, :]

    diff = torch.stack((diff_h, diff_w), dim=-1)
    _, indices = torch.unique(diff.view(-1, 2), return_inverse=True, dim=0)
    return indices.view(z_shape[0] * z_shape[1], x_shape[0] * x_shape[1])


def generate_2d_concatenated_self_attention_relative_positional_encoding_index(z_shape, x_shape):
    '''
        z_shape: (z_h, z_w)
        x_shape: (x_h, x_w)
    '''
    z_2d_index_h, z_2d_index_w = torch.meshgrid(torch.arange(z_shape[0]), torch.arange(z_shape[1]))
    x_2d_index_h, x_2d_index_w = torch.meshgrid(torch.arange(x_shape[0]), torch.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    concatenated_2d_index_h = torch.cat((z_2d_index_h, x_2d_index_h))
    concatenated_2d_index_w = torch.cat((z_2d_index_w, x_2d_index_w))

    diff_h = concatenated_2d_index_h[:, None] - concatenated_2d_index_h[None, :]
    diff_w = concatenated_2d_index_w[:, None] - concatenated_2d_index_w[None, :]

    z_len = z_shape[0] * z_shape[1]
    x_len = x_shape[0] * x_shape[1]
    a = torch.empty((z_len + x_len), dtype=torch.int64)
    a[:z_len] = 0
    a[z_len:] = 1
    b=a[:, None].repeat(1, z_len + x_len)
    c=a[None, :].repeat(z_len + x_len, 1)

    diff = torch.stack((diff_h, diff_w, b, c), dim=-1)
    _, indices = torch.unique(diff.view((z_len + x_len) * (z_len + x_len), 4), return_inverse=True, dim=0)
    return indices.view((z_len + x_len), (z_len + x_len))


def generate_2d_concatenated_cross_attention_relative_positional_encoding_index(z_shape, x_shape):
    '''
        z_shape: (z_h, z_w)
        x_shape: (x_h, x_w)
    '''
    z_2d_index_h, z_2d_index_w = torch.meshgrid(torch.arange(z_shape[0]), torch.arange(z_shape[1]))
    x_2d_index_h, x_2d_index_w = torch.meshgrid(torch.arange(x_shape[0]), torch.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    concatenated_2d_index_h = torch.cat((z_2d_index_h, x_2d_index_h))
    concatenated_2d_index_w = torch.cat((z_2d_index_w, x_2d_index_w))

    diff_h = x_2d_index_h[:, None] - concatenated_2d_index_h[None, :]
    diff_w = x_2d_index_w[:, None] - concatenated_2d_index_w[None, :]

    z_len = z_shape[0] * z_shape[1]
    x_len = x_shape[0] * x_shape[1]

    a = torch.empty(z_len + x_len, dtype=torch.int64)
    a[: z_len] = 0
    a[z_len:] = 1
    c = a[None, :].repeat(x_len, 1)

    diff = torch.stack((diff_h, diff_w, c), dim=-1)
    _, indices = torch.unique(diff.view(x_len * (z_len + x_len), 3), return_inverse=True, dim=0)
    return indices.view(x_len, (z_len + x_len))


class RelativePosition2DEncoder(nn.Module):
    def __init__(self, num_heads, embed_size):
        super(RelativePosition2DEncoder, self).__init__()
        self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads, embed_size)))
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, attn_rpe_index):
        '''
            Args:
                attn_rpe_index (torch.Tensor): (*), any shape containing indices, max(attn_rpe_index) < embed_size
            Returns:
                torch.Tensor: (1, num_heads, *)
        '''
        return self.relative_position_bias_table[:, attn_rpe_index].unsqueeze(0)

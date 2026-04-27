import torch
import math
from torch import Tensor
import numpy as np
from pytorch3d.ops import knn_points

def normalize(x: Tensor, offset: Tensor, scale: Tensor, ratio: Tensor, ratio_dim: int = -1) -> Tensor:
    x_normalized = x.clone()
    x_normalized -= offset
    x_normalized /= scale
    x_normalized[..., ratio_dim] *= ratio
    return x_normalized

def denormalize(x, offset, scale, ratio, ratio_dim=-1):
    x_denormalized = x.clone()
    x_denormalized[..., ratio_dim] /= ratio
    x_denormalized *= scale

    x_denormalized += offset
    return x_denormalized

def _query_weights_smpl(x, smpl_verts, smpl_weights, resolution_dhw):
    # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py
    dist, idx, _ = knn_points(x, smpl_verts.detach(), K=30) # [B, N, 30]
    dist = dist.sqrt().clamp_(0.0001, 1.0)
    expanded_smpl_weights = smpl_weights.unsqueeze(2).expand(-1, -1, idx.shape[2], -1) # [B, N, 30, J]
    weights = expanded_smpl_weights.gather(1, idx.unsqueeze(-1).expand(-1, -1, -1, expanded_smpl_weights.shape[-1])) # [B, N, 30, J]

    ws = 1.0 / dist
    ws = ws / ws.sum(-1, keepdim=True)
    weights = (ws[..., None] * weights).sum(-2)

    b = x.shape[0]
    c = smpl_weights.shape[-1]
    d, h, w = resolution_dhw
    weights = weights.permute(0, 2, 1).reshape(b, c, d, h, w)
    for _ in range(30):
        mean = (
            weights[:, :, 2:, 1:-1, 1:-1]
            + weights[:, :, :-2, 1:-1, 1:-1]
            + weights[:, :, 1:-1, 2:, 1:-1]
            + weights[:, :, 1:-1, :-2, 1:-1]
            + weights[:, :, 1:-1, 1:-1, 2:]
            + weights[:, :, 1:-1, 1:-1, :-2]
        ) / 6.0
        weights[:, :, 1:-1, 1:-1, 1:-1] = (
            weights[:, :, 1:-1, 1:-1, 1:-1] - mean
        ) * 0.7 + mean
        sums = weights.sum(1, keepdim=True)
        weights = weights / sums
    return weights.detach()

def build_grid(B, device, resolution_dhw=[64, 64, 32], short_dim_dhw=1,  # 0 is d, corresponding to z
    long_dim_dhw=0):
    d, h, w = resolution_dhw
    x_range = (
        (torch.linspace(-1, 1, steps=d, device=device))
        .view(1, d, 1, 1)
        .expand(1, d, h, w)
    )
    y_range = (
        (torch.linspace(-1, 1, steps=w, device=device))
        .view(1, 1, 1, w)
        .expand(1, d, h, w)
    )
    z_range = (
        (torch.linspace(-1, 1, steps=h, device=device))
        .view(1, 1, h, 1)
        .expand(1, d, h, w)
    )
    grid = (
        torch.cat((x_range, y_range, z_range), dim=0)
        .reshape(1, 3, -1)
        .permute(0, 2, 1)
    )
    grid = grid.expand(B, -1, -1)

    return grid

def denormalize(x, offset, scale, ratio, ratio_dim=-1):
    x_denormalized = x.clone()
    x_denormalized[..., ratio_dim] /= ratio
    x_denormalized *= scale

    x_denormalized += offset
    return x_denormalized

def build_voxel_grid(vtx, vtx_features=None, grid=None, resolution_dhw=[64, 64, 32], short_dim_dhw=1,  # 0 is d, corresponding to z
    long_dim_dhw=0, use_grid=True):
    # vtx B,N,3, vtx_features: B,N,J
    # d-z h-y w-x; human is facing z; dog is facing x, z is upward, should compress on y
    # d-x h-z w-y; human is facing x; z is upward, should compress on y
    B = vtx.shape[0]
    # * Prepare Grid
    device = vtx.device
    ratio = torch.Tensor([resolution_dhw[long_dim_dhw] / resolution_dhw[short_dim_dhw]]).squeeze()
    ratio_dim = -1 - short_dim_dhw
    # assert vtx.shape[0] == vtx_features.shape[0], "Batch size mismatch"
    if grid is None:
        grid = build_grid(B, device, resolution_dhw, short_dim_dhw, long_dim_dhw)
    
    gt_bbox_min = (vtx.min(dim=1).values).to(device)
    gt_bbox_max = (vtx.max(dim=1).values).to(device)
    offset = (gt_bbox_min + gt_bbox_max) * 0.5 
    global_scale = torch.Tensor([1.0]).squeeze()
    scale = (
        (gt_bbox_max - gt_bbox_min).max(dim=-1).values / 2 * global_scale 
    ).unsqueeze(-1) #类似于一个半径

    corner = torch.ones_like(offset) * scale
    corner[:, ratio_dim] /= ratio
    min_vert = (offset - corner).reshape(-1, 1, 3)  
    # max_vert = (offset + corner).reshape(-1, 1, 3)
    # bbox = torch.cat([min_vert, max_vert], dim=1)

    scale = scale.unsqueeze(1) # [B, 1, 1]
    offset = offset.unsqueeze(1) # [B, 1, 3]
    
    # grid_denorm[..., 0] *= 0.43

    lengths = torch.ones_like(offset) * scale  # [B, 3]
    lengths[0, 0, ratio_dim] /= ratio
    voxel_size = lengths * 2 / torch.tensor(resolution_dhw, device=lengths.device)  # [B,3]

    # 输出第一个样本的体素分辨率
    # print("voxel_size (x, y, z):", voxel_size[0,0])
    if use_grid:
        grid_denorm = denormalize(grid, offset, scale, ratio, ratio_dim)  # grid_denorm is in the same scale as the canonical body
        return grid_denorm[0], min_vert, voxel_size[0,0]#, [scale, ratio_dim, ratio], 
    else:
        return None, min_vert, voxel_size[0,0]#, [scale, ratio_dim, ratio], 

# import torch

def smpl_to_voxel(verts, resolution=64, padding_ratio=1.2):
    """
    :param verts: [N, 3] or [B, N, 3] 
    :param resolution: voxel 
    :param padding_ratio: 
    :return: voxel_points [B, resolution, resolution, resolution, 3]
    """
    if verts.dim() == 2:
        verts = verts.unsqueeze(0)  # [1, N, 3]

    B = verts.shape[0]
    device = verts.device

    # 计算包围盒
    vmin = verts.min(dim=1).values  # [B, 3]
    vmax = verts.max(dim=1).values  # [B, 3]
    center = (vmin + vmax) / 2
    half_size = (vmax - vmin) / 2 * padding_ratio

    min_corner = center - half_size
    max_corner = center + half_size


    xs = torch.linspace(0, 1, resolution, device=device).unsqueeze(0)  # [1, R]
    ys = torch.linspace(0, 1, resolution, device=device).unsqueeze(0)
    zs = torch.linspace(0, 1, resolution, device=device).unsqueeze(0)

    xs = min_corner[:, 0:1] + xs * (max_corner[:, 0:1] - min_corner[:, 0:1])
    ys = min_corner[:, 1:2] + ys * (max_corner[:, 1:2] - min_corner[:, 1:2])
    zs = min_corner[:, 2:3] + zs * (max_corner[:, 2:3] - min_corner[:, 2:3])

    xv, yv, zv = torch.meshgrid(xs[0], ys[0], zs[0], indexing="ij")
    # grid = torch.stack([xv, yv, zv], dim=-1)  # [R, R, R, 3]
    # grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, R, R, R, 3]
    grid = torch.stack([xv, yv, zv], dim=-1).reshape(-1, 3)  # [R^3, 3]

    # 映射到每个 batch 的实际坐标
    voxel_points = []
    for b in range(B):
        real_coords = min_corner[b] + grid * (max_corner[b] - min_corner[b])
        voxel_points.append(real_coords)
    voxel_points = torch.stack(voxel_points, dim=0)  # [B, R^3, 3]

    if voxel_points.shape[0] == 1:
        voxel_points = voxel_points.squeeze(0)  # [R^3, 3]

    return voxel_points



# # 示例
# verts = torch.rand(6890, 3) * 2 - 1  # 模拟 SMPL 顶点
# voxel_points = smpl_to_voxel(verts, resolution=32)
# print(voxel_points.shape)  # torch.Size([1, 32, 32, 32, 3])

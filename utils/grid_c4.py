import os, path, glob
import torch
import math
from torch import Tensor
import numpy as np
from pytorch3d.ops import knn_points, sample_farthest_points, knn_gather
import torch.nn.functional as F
from typing import Union, Sequence, Tuple
from utils.o3d_vis import visualize_point_cloud, bcpdreg
from models.gaussians.basics import *
from third_party.smplx.smplx import SMPLLayer
from third_party.smplx.smplx.utils import SMPLOutput
from third_party.smplx.smplx.lbs import vertices2joints, batch_rigid_transform
from scipy.spatial.transform import Rotation 
from utils.lib import pointnet2_utils as pointutils
import matplotlib.pyplot as plt
import open3d as o3d
from utils.geometry import transform_points
import re

feet_vids = [
    3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 
    3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 
    3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188,
    3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3207, 3208, 3209, 3210,
        3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222,
        3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234,
        3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246,
        3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258,
        3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270,
        3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282,
        3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294,
        3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306,
        3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318,
        3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335,
        3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347,
        3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359,
        3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371,
        3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383,
        3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395,
        3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407,
        3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419,
        3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431,
        3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443,
        3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455,
        3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467,
        3468, 3469, 6550, 6551, 6552, 6553, 6554, 6555, 6556, 6557, 6558, 6559, 
        6560, 6561, 6562, 6563, 6564, 6565, 6566, 6567, 6568, 6569, 6570, 6571, 
        6572, 6573, 6574, 6575, 6576, 6577, 6578, 6579, 6580, 6581, 6582, 6583, 
        6584, 6585, 6586, 6587, 6588, 6589, 6590, 6591, 6592, 6593, 6594, 6595, 
        6596, 6597, 6598, 6599, 6600, 6601, 6602, 6603, 6604, 6606, 6607, 6608,
        6609, 6610, 6611, 6612, 6613, 6614, 6615, 6616, 6617, 6618, 6619, 6620,
        6621, 6622, 6623, 6624, 6625, 6626, 6627, 6628, 6629, 6630, 6631, 6632,
        6633, 6634, 6635, 6636, 6637, 6638, 6639, 6640, 6641, 6642, 6643, 6644,
        6645, 6646, 6647, 6648, 6649, 6650, 6651, 6652, 6653, 6654, 6655, 6656,
        6657, 6658, 6659, 6660, 6661, 6662, 6663, 6664, 6665, 6666, 6667, 6668,
        6669, 6670, 6671, 6672, 6673, 6674, 6675, 6676, 6677, 6678, 6679, 6680,
        6681, 6682, 6683, 6684, 6685, 6686, 6687, 6688, 6689, 6690, 6691, 6692,
        6693, 6694, 6695, 6696, 6697, 6698, 6699, 6700, 6701, 6702, 6703, 6704,
        6705, 6706, 6707, 6708, 6709, 6710, 6711, 6712, 6713, 6714, 6715, 6716,
        6717, 6718, 6724, 6725, 6726, 6727, 6728, 6729, 6730, 6731, 6732, 6733,
        6734, 6735, 6736, 6737, 6738, 6739, 6740, 6741, 6742, 6743, 6744, 6745,
        6746, 6747, 6748, 6749, 6750, 6751, 6752, 6753, 6754, 6755, 6756, 6757,
        6758, 6759, 6760, 6761, 6762, 6763, 6764, 6765, 6766, 6767, 6768, 6769,
        6770, 6771, 6772, 6773, 6774, 6775, 6776, 6777, 6778, 6779, 6780, 6781,
        6782, 6783, 6784, 6785, 6786, 6787, 6788, 6789, 6790, 6791, 6792, 6793,
        6794, 6795, 6796, 6797, 6798, 6799, 6800, 6801, 6802, 6803, 6804, 6805,
        6806, 6807, 6808, 6809, 6810, 6811, 6812, 6813, 6814, 6815, 6816, 6817,
        6818, 6819, 6820, 6821, 6822, 6823, 6824, 6825, 6826, 6827, 6828, 6829,
        6830, 6831, 6832, 6833, 6834, 6835, 6836, 6837, 6838, 6839, 6840, 6841,
        6842, 6843, 6844, 6845, 6846, 6847, 6848, 6849, 6850, 6851, 6852, 6853,
        6854, 6855, 6856, 6857, 6858, 6859, 6860, 6861, 6862, 6863, 6864, 6865,
        6866, 6867, 6868, 6869
]



from models.human_body import get_predefined_human_rest_pose, init_xyz_on_mesh, init_qso_on_mesh
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    quaternion_invert
)

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

def build_grid(B, device, resolution_dhw=[64, 64, 32], short_dim_dhw=1, long_dim_dhw=0):
    d, h, w = resolution_dhw
    x_range = (
        (torch.linspace(-1, 1, steps=d, device=device))
        .view(1, d, 1, 1)
        .expand(1, d, h, w)
    )
    z_range = (
        (torch.linspace(-1, 1, steps=w, device=device))
        .view(1, 1, 1, w)
        .expand(1, d, h, w)
    )
    y_range = (
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

def smpl_to_voxel(verts, resolution=64, padding_ratio=1.2):
    """
    :param verts: [N, 3] or [B, N, 3] 
    :param resolution: voxel 
    :param padding_ratio: 
    :return: voxel_points [B, resolution^3, 3]
    """
    if verts.dim() == 2:
        verts = verts.unsqueeze(0)  # [1, N, 3]

    B = verts.shape[0]
    device = verts.device

    vmin = verts.min(dim=1).values  # [B, 3]
    vmax = verts.max(dim=1).values  # [B, 3]
    center = (vmin + vmax) / 2
    half_size = (vmax - vmin) / 2 * padding_ratio

    min_corner = center - half_size
    max_corner = center + half_size

    xs = torch.linspace(0, 1, resolution, device=device)
    ys = torch.linspace(0, 1, resolution, device=device)
    zs = torch.linspace(0, 1, resolution, device=device)
    
    voxel_points = []
    for b in range(B):
        x_coords = min_corner[b, 0] + xs * (max_corner[b, 0] - min_corner[b, 0])
        y_coords = min_corner[b, 1] + ys * (max_corner[b, 1] - min_corner[b, 1])
        z_coords = min_corner[b, 2] + zs * (max_corner[b, 2] - min_corner[b, 2])
        
        xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        grid = torch.stack([xv, yv, zv], dim=-1).reshape(-1, 3)  # [R^3, 3]
        voxel_points.append(grid)
    
    voxel_points = torch.stack(voxel_points, dim=0)  # [B, R^3, 3]

    if B == 1:
        voxel_points = voxel_points.squeeze(0)  # [R^3, 3]

    return voxel_points


def build_cot_laplacian_torch(V, F):
    device = V.device
    N = V.shape[0]

    vi = V[F[:,0]]
    vj = V[F[:,1]]
    vk = V[F[:,2]]

    cot_alpha = torch.sum((vj-vi)*(vk-vi), dim=1) / (
        torch.norm(torch.cross(vj-vi, vk-vi), dim=1) + 1e-8
    )
    cot_beta = torch.sum((vi-vj)*(vk-vj), dim=1) / (
        torch.norm(torch.cross(vi-vj, vk-vj), dim=1) + 1e-8
    )
    cot_gamma = torch.sum((vi-vk)*(vj-vk), dim=1) / (
        torch.norm(torch.cross(vi-vk, vj-vk), dim=1) + 1e-8
    )

    w_ij = 0.5 * cot_gamma
    w_jk = 0.5 * cot_alpha
    w_ki = 0.5 * cot_beta

    I = torch.cat([F[:,0], F[:,1], F[:,1], F[:,2], F[:,2], F[:,0]], dim=0)
    J = torch.cat([F[:,1], F[:,0], F[:,2], F[:,1], F[:,0], F[:,2]], dim=0)
    W = torch.cat([-w_ij, -w_ij, -w_jk, -w_jk, -w_ki, -w_ki], dim=0)

    diag_entries = torch.zeros(N, device=device)
    diag_entries.index_add_(0, F[:,0], w_ij + w_ki)
    diag_entries.index_add_(0, F[:,1], w_ij + w_jk)
    diag_entries.index_add_(0, F[:,2], w_jk + w_ki)

    I_diag = torch.arange(N, device=device)
    J_diag = torch.arange(N, device=device)
    W_diag = diag_entries

    I_all = torch.cat([I, I_diag])
    J_all = torch.cat([J, J_diag])
    W_all = torch.cat([W, W_diag])

    L = torch.sparse_coo_tensor(torch.stack([I_all, J_all], dim=0), W_all, (N, N))
    return L.coalesce()

def area_regularization(V, F, V_orig):
    """
    V: (N,3) 
    F: (M,3) 
    V_orig: (N,3) 
    """
    vi, vj, vk = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    cross_new = torch.cross(vj - vi, vk - vi, dim=1)
    area_new = 0.5 * torch.norm(cross_new, dim=1)

    vi0, vj0, vk0 = V_orig[F[:,0]], V_orig[F[:,1]], V_orig[F[:,2]]
    cross_orig = torch.cross(vj0 - vi0, vk0 - vi0, dim=1)
    area_orig = 0.5 * torch.norm(cross_orig, dim=1)

    return ((area_new - area_orig) ** 2).sum()

def compute_grid_knn_distance_smooth(grids, smpl_verts, inner_verts=None, grid_shape=None, k=8, eps=1e-8):
    '''
    grids: (B, N, 3) or (N, 3)
    smpl_verts: (M, 3) or (B, M, 3)
    '''
    if grids.dim() == 4:  # (Dx, Dy, Dz, 3)
        Dx, Dy, Dz, _ = grids.shape
        grids_flat = grids.view(-1, 3).unsqueeze(0)  # (1, Nvoxel, 3)
    elif grids.dim() == 3:  # (B, N, 3)
        grids_flat = grids
    else:  # (N, 3)
        grids_flat = grids.unsqueeze(0)
    
    if smpl_verts.dim() == 2:  # (M, 3)
        smpl_verts = smpl_verts.unsqueeze(0)  # (1, M, 3)
    
    # dists, knn_idx, _ = knn_points(grids_flat, smpl_verts.detach(), K=k)
    dists, knn_idx = pointutils.knn(k, grids_flat.contiguous(), smpl_verts.contiguous())
    dists[dists < 1e-6] = 1e-6

    B, N, _ = grids_flat.shape
    _, M, _ = smpl_verts.shape
    
    knn_verts = torch.gather(
        smpl_verts.expand(B, -1, -1), 1, 
        knn_idx.view(B, -1).unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
    ).view(B, N, k, 3)

    knn_inner_verts = torch.gather(
        inner_verts.expand(B, -1, -1), 1, 
        knn_idx.view(B, -1).unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
    ).view(B, N, k, 3)
    
    if k==1:
        smooth_vectors = knn_verts.squeeze(-2)  - grids_flat
        smooth_with_inner_vectors = knn_verts.squeeze(-2)  - grids_flat - knn_inner_verts.squeeze(-2)
    else:
        motion_vectors = knn_verts - grids_flat.unsqueeze(-2)
        motion_with_inner_vectors = knn_verts - grids_flat.unsqueeze(-2) + inner_verts # (B, N, k, 3) 
        weights = 1.0 / (dists + eps)  # (B, N, k)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        smooth_vectors = (weights.unsqueeze(-1) * motion_vectors).sum(dim=-2)  # (B, N, 3)
        smooth_with_inner_vectors = (weights.unsqueeze(-1) * motion_with_inner_vectors).sum(dim=-2)  # (B, N, 3)
    # smooth_vectors = smooth_vectors[:,:,[0,2,1]]
    smooth_vectors = smooth_vectors.permute(0, 2, 1).reshape(B, 3, *grid_shape)
    smooth_with_inner_vectors = smooth_with_inner_vectors.permute(0, 2, 1).reshape(B, 3, *grid_shape)

    for _ in range(0):
        smooth_vectors[:, :, 1:-1, 1:-1, 1:-1] = (
            smooth_vectors[:, :, 2:, 1:-1, 1:-1]
            + smooth_vectors[:, :, :-2, 1:-1, 1:-1]
            + smooth_vectors[:, :, 1:-1, 2:, 1:-1]
            + smooth_vectors[:, :, 1:-1, :-2, 1:-1]
            + smooth_vectors[:, :, 1:-1, 1:-1, 2:]
            + smooth_vectors[:, :, 1:-1, 1:-1, :-2]
        ) / 6.0
        # smooth_vectors[:, :, 1:-1, 1:-1, 1:-1] = (
        #     smooth_vectors[:, :, 1:-1, 1:-1, 1:-1] - mean
        # ) * 0.0 + mean
        # sums = smooth_vectors.sum(1, keepdim=True)
        # smooth_vectors = smooth_vectors / sums

        smooth_with_inner_vectors[:, :, 1:-1, 1:-1, 1:-1] = (
            smooth_with_inner_vectors[:, :, 2:, 1:-1, 1:-1]
            + smooth_with_inner_vectors[:, :, :-2, 1:-1, 1:-1]
            + smooth_with_inner_vectors[:, :, 1:-1, 2:, 1:-1]
            + smooth_with_inner_vectors[:, :, 1:-1, :-2, 1:-1]
            + smooth_with_inner_vectors[:, :, 1:-1, 1:-1, 2:]
            + smooth_with_inner_vectors[:, :, 1:-1, 1:-1, :-2]
        ) / 6.0
        # smooth_with_inner_vectors[:, :, 1:-1, 1:-1, 1:-1] = (
        #     smooth_with_inner_vectors[:, :, 1:-1, 1:-1, 1:-1] - mean
        # ) * 0.1 + mean
        # sums = smooth_with_inner_vectors.sum(1, keepdim=True)
        # smooth_with_inner_vectors = smooth_with_inner_vectors / sums

    return smooth_vectors, smooth_with_inner_vectors, dists, knn_idx

def build_voxel_grid(vtx, vtx_features=None, grid=None, resolution_dhw=[64, 64, 32], 
                    short_dim_dhw=1, long_dim_dhw=0, use_grid=True):
    B = vtx.shape[0]
    device = vtx.device
    ratio = torch.tensor(resolution_dhw[long_dim_dhw] / resolution_dhw[short_dim_dhw], device=device)
    ratio_dim = -1 - short_dim_dhw

    if grid is None:
        grid = build_grid(B, device, resolution_dhw, short_dim_dhw, long_dim_dhw)
    
    gt_bbox_min = vtx.min(dim=1).values.to(device)
    gt_bbox_max = vtx.max(dim=1).values.to(device)
    offset = (gt_bbox_min + gt_bbox_max) * 0.5 
    global_scale = 1.0
    scale = (gt_bbox_max - gt_bbox_min).max(dim=-1).values / 2 * global_scale
    scale = scale.unsqueeze(-1)

    corner = torch.ones_like(offset) * scale
    corner[:, ratio_dim] /= ratio
    min_vert = (offset - corner).reshape(-1, 1, 3)

    scale = scale.unsqueeze(1)
    offset = offset.unsqueeze(1)
    
    lengths = torch.ones_like(offset) * scale
    lengths[0, 0, ratio_dim] /= ratio
    voxel_size = lengths * 2 / torch.tensor(resolution_dhw, device=lengths.device, dtype=lengths.dtype)

    if use_grid:
        grid_denorm = denormalize(grid, offset, scale, ratio, ratio_dim)
        return grid_denorm[0], min_vert, voxel_size[0, 0]
    else:
        return None, min_vert, voxel_size[0, 0]

def voxel_knn_local(src: torch.Tensor,
                   grid_min: Union[torch.Tensor, Sequence[float]],
                   voxel_size: Union[float, torch.Tensor, Sequence[float]],
                   grid_shape: Tuple[int, int, int],
                   K: int = 1,
                   neighbor_range: Union[int, Sequence[int]] = 1):
    assert src.ndim == 3 and src.shape[-1] == 3, "src must be (B, N, 3)"
    B, N, _ = src.shape
    D, H, W = grid_shape
    device = src.device
    dtype = src.dtype

    grid_min = torch.as_tensor(grid_min, device=device, dtype=dtype)
    if grid_min.ndim == 1:
        grid_min = grid_min.view(1, 1, 3).expand(B, 1, 3)
    elif grid_min.ndim == 2:
        grid_min = grid_min.view(B, 1, 3)

    voxel_size = torch.as_tensor(voxel_size, device=device, dtype=dtype)
    if voxel_size.ndim == 0:
        voxel_size = voxel_size.expand(3).view(1, 1, 3).expand(B, 1, 3)
    elif voxel_size.ndim == 1:
        voxel_size = voxel_size.view(1, 1, 3).expand(B, 1, 3)
    elif voxel_size.ndim == 2:
        voxel_size = voxel_size.view(B, 1, 3)

    idx = torch.floor((src - grid_min) / voxel_size).long()
    idx[..., 0].clamp_(0, D - 1)
    idx[..., 1].clamp_(0, H - 1)
    idx[..., 2].clamp_(0, W - 1)

    if isinstance(neighbor_range, int):
        rx = ry = rz = neighbor_range
    else:
        rx, ry, rz = map(int, neighbor_range)

    ox = torch.arange(-rx, rx + 1, device=device)
    oy = torch.arange(-ry, ry + 1, device=device)
    oz = torch.arange(-rz, rz + 1, device=device)
    offsets = torch.stack(torch.meshgrid(ox, oy, oz, indexing="ij"), dim=-1).view(-1, 3)
    M = offsets.shape[0]

    idx_expanded = idx.unsqueeze(-2) + offsets.view(1, 1, M, 3)
    idx_expanded[..., 0].clamp_(0, D - 1)
    idx_expanded[..., 1].clamp_(0, H - 1)
    idx_expanded[..., 2].clamp_(0, W - 1)

    voxel_centers = grid_min.unsqueeze(2) + (idx_expanded.to(dtype) + 0.5) * voxel_size.unsqueeze(2)

    diff = voxel_centers - src.unsqueeze(-2)
    dist2 = (diff * diff).sum(dim=-1)

    K_eff = min(K, M)
    knn_idx = dist2.topk(k=K_eff, largest=False).indices

    neighbors = torch.gather(
        voxel_centers,
        dim=2,
        index=knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    )

    return neighbors

def get_cube_neighbor_indices(voxel_idx, grid_shape):
    device = voxel_idx.device
    offsets = torch.stack(torch.meshgrid(
        torch.arange(-1, 2, device=device),
        torch.arange(-1, 2, device=device),
        torch.arange(-1, 2, device=device),
        indexing='ij'
    ), -1).reshape(-1, 3)
    
    neighbor_idx = voxel_idx[:, None, :] + offsets[None, :, :]
    
    for d in range(3):
        neighbor_idx[:, :, d] = neighbor_idx[:, :, d].clamp(0, grid_shape[d] - 1)
    
    return neighbor_idx

def query_neighbors_dense_grid(neighbor_idx, occupancy, id_map, coord_map, distance=None):
    NB, Ncube = neighbor_idx.shape[:2]
    i, j, k = neighbor_idx[:, :, 0], neighbor_idx[:, :, 1], neighbor_idx[:, :, 2]
    
    neighbor_occupy = occupancy[i, j, k]
    neighbor_id = id_map[i, j, k]
    neighbor_coord = coord_map[i, j, k]
    
    neighbor_dist = None
    if distance is not None:
        neighbor_dist = distance[i, j, k]
    
    return neighbor_occupy, neighbor_id, neighbor_coord, neighbor_dist

def get_grid_parameters(all_pts, resolution_dhw=[64, 64, 32], margin=1e-3, padding_factor=1.2):
    device = all_pts.device
    d, h, w = resolution_dhw
    
    grid_min = all_pts.min(dim=0)[0] - margin
    grid_max = all_pts.max(dim=0)[0] + margin
    
    center = (grid_min + grid_max) / 2
    extent = (grid_max - grid_min) * padding_factor / 2
    grid_min = center - extent
    grid_max = center + extent

    grid_range = grid_max - grid_min
    

    # build_grid: (x, y, z) = (d_coord, w_coord, h_coord)
    grid_size = torch.stack([
        grid_range[0] / d,  # x -> d
        grid_range[1] / w,  # y -> w
        grid_range[2] / h   # z -> h
    ])
    
    return grid_min, grid_max, grid_size

def world_to_grid_coords(world_coords, grid_min, grid_size):
    return ((world_coords - grid_min) / grid_size).long()

def grid_to_world_coords(grid_indices, grid_min, grid_size):
    return grid_min + (grid_indices.float() + 0.5) * grid_size

def world_to_normalized_coords(world_coords, grid_min, grid_max):
    return 2.0 * (world_coords - grid_min) / (grid_max - grid_min) - 1.0


def get_grid_parameters_corrected(all_pts, resolution_dhw=[64, 64, 32], margin=1e-3, padding_factor=1.0):
    device = all_pts.device
    d, h, w = resolution_dhw  # d=64, h=64, w=32
    
    grid_min = all_pts.min(dim=0)[0] - margin
    grid_max = all_pts.max(dim=0)[0] + margin
    
    center = (grid_min + grid_max) / 2
    extent = (grid_max - grid_min) * padding_factor / 2
    grid_min = center - extent
    grid_max = center + extent

    grid_range = grid_max - grid_min
    

    # x -> d (64)
    # y -> h (64)  
    # z -> w (32)
    grid_size = torch.stack([
        grid_range[0] / d,  # x -> d
        grid_range[1] / h,  # y -> h
        grid_range[2] / w   # z -> w
    ])
    
    return grid_min, grid_max, grid_size

def create_dense_grid_with_knn_corrected(A, B, smpl_verts, grid_world, resolution_dhw=[64, 64, 32]):
    Dx, Dy, Dz = resolution_dhw  # d=64, h=64, w=32
    device = A.device

    all_pts = torch.cat([A, B, smpl_verts], dim=0)
    grid_min, grid_max, grid_size = get_grid_parameters_corrected(all_pts, resolution_dhw)
    
    print(f"Grid size: {grid_size}")
    print(f"Grid min: {grid_min}")
    print(f"Grid max: {grid_max}")
    
    # [Dx, Dy, Dz]
    occupancy = torch.zeros((Dx, Dy, Dz), dtype=torch.bool, device=device)
    id_map = torch.full((Dx, Dy, Dz), -1, dtype=torch.long, device=device)
    coord_map = torch.zeros((Dx, Dy, Dz, 3), dtype=A.dtype, device=device)
    
    # 
    idx_A = world_to_grid_coords(A, grid_min, grid_size)
    
    valid_mask_A = ((idx_A >= 0) & (idx_A < torch.tensor([Dx, Dy, Dz], device=device))).all(dim=1)
    idx_A = idx_A[valid_mask_A]
    A_valid = A[valid_mask_A]
    ids = torch.arange(A_valid.shape[0], device=device)
    
    occupancy[idx_A[:, 0], idx_A[:, 1], idx_A[:, 2]] = True
    id_map[idx_A[:, 0], idx_A[:, 1], idx_A[:, 2]] = ids
    coord_map[idx_A[:, 0], idx_A[:, 1], idx_A[:, 2]] = A_valid

    smooth_vectors, _, _ = compute_grid_knn_distance_smooth(
        grid_world, smpl_verts, k=1, grid_shape=resolution_dhw
    )
    
    B_normalized = world_to_normalized_coords(B, grid_min, grid_max)
    B_normalized = B_normalized.clamp(-1, 1)
    
    B_sample_coords = B_normalized[:, None, None, :]
    
    sampled_motions_B = F.grid_sample(
        smooth_vectors,  # [1, 3, Dx, Dy, Dz] = [1, 3, d, h, w]
        B_sample_coords.unsqueeze(0),  # [1, N, 1, 1, 3]
        mode='bilinear', 
        align_corners=True, 
        padding_mode="border",
    )
    
    sampled_motions_B = sampled_motions_B.squeeze().transpose(0, 1)  # [N, 3]
    
    B_canonical = B + sampled_motions_B
    
    idx_B = world_to_grid_coords(B_canonical, grid_min, grid_size)
    valid_mask_B = ((idx_B >= 0) & (idx_B < torch.tensor([Dx, Dy, Dz], device=device))).all(dim=1)
    
    valid_idx_B = idx_B[valid_mask_B]
    if valid_idx_B.shape[0] > 0:
        valid_mask_occupied = occupancy[valid_idx_B[:, 0], valid_idx_B[:, 1], valid_idx_B[:, 2]]
        
        if valid_mask_occupied.any():
            final_valid_idx = valid_idx_B[valid_mask_occupied]
            B_valid_canonical = B_canonical[valid_mask_B][valid_mask_occupied]
            
            motion_A = coord_map[final_valid_idx[:, 0], final_valid_idx[:, 1], final_valid_idx[:, 2]] - B_valid_canonical
            
            known_idx = final_valid_idx.float()
            known_motion = motion_A
            
            grid_flat = torch.stack(torch.meshgrid(
                torch.arange(Dx, device=device),  
                torch.arange(Dy, device=device),  
                torch.arange(Dz, device=device),  
                indexing='ij'
            ), -1).reshape(-1, 3).float()
            
            if known_idx.shape[0] > 0:
                dists, idx, _ = knn_points(grid_flat[None], known_idx[None], K=1)
                nn_idx = idx[0, :, 0]
                motion_full = known_motion[nn_idx].view(Dx, Dy, Dz, 3)
                min_dists = dists[0, :, 0].view(Dx, Dy, Dz)
            else:
                motion_full = torch.zeros((Dx, Dy, Dz, 3), device=device)
                min_dists = torch.ones((Dx, Dy, Dz), device=device) * float('inf')
        else:
            motion_full = torch.zeros((Dx, Dy, Dz, 3), device=device)
            min_dists = torch.ones((Dx, Dy, Dz), device=device) * float('inf')
    else:
        motion_full = torch.zeros((Dx, Dy, Dz, 3), device=device)
        min_dists = torch.ones((Dx, Dy, Dz), device=device) * float('inf')
    
    return occupancy, id_map, coord_map, min_dists, motion_full, smooth_vectors, grid_world, grid_min, grid_size

def gaussian_kernel(a, b, beta=20.0):
    # a = a.half()  
    # b = b.half()
    sq_dist = torch.sum((a[:, None, :] - b[None, :, :]) ** 2, dim=2, dtype=b.dtype)
    return torch.exp(-sq_dist / (2 * beta ** 2)).float()


def compute_v_vector_wise(
        z:   torch.Tensor,      # [K, D] 
        Y:   torch.Tensor,      # [M, D]
        w_m: torch.Tensor,      # [M, D]
        sigma: float,
        *,
        target_device: Union[str, torch.device] = "cuda:0",   
        chunk_size: int = 65_536                       
    ) -> torch.Tensor:
    K, D  = z.shape
    # v_buf = torch.empty((K, D), device=tgt_dev, dtype=torch.float32)
    Y_norm2   = (Y ** 2).sum(1)          # (M,)
    v_list = []
    for beg in range(0, K, chunk_size):
        end   = min(beg + chunk_size, K)
        z_ch  = z[beg:end]                    # (B, D)

       

        z_norm2  = (z_ch ** 2).sum(1, keepdim=True)   # (B,1)
        cross    = z_ch @ Y.T                       # (B,M)

        d2 = z_norm2 + Y_norm2 - 2 * cross            # (B,M)
        G  = torch.exp(-d2 / (2*sigma*sigma))
        v_chunk_gpu = G @ w_m                       # (B, D)
        # v_chunk_gpu = torch.matmul(G, w_m_gpu)          # (B, D)
        v_list.append(v_chunk_gpu)
    v = torch.cat(v_list, dim=0)  # [K, D]

    return v

def get_predictions(sigma: float, beta: float, means_: torch.Tensor, srcX: torch.Tensor, srcY: torch.Tensor, lmbd=1e-5) -> torch.Tensor:
        """
        Forward pass of the CPD Gaussians model.
        """
        beta = 20.0
        srcX = srcY
        srcX = means_
        
        # Delta_X = srcY - srcX
        G_xx = gaussian_kernel(srcX, srcX, beta=beta)  # (B,B)
        G_xx_reg = G_xx + lmbd * torch.eye(srcX.shape[0]).to(srcY.device)

        #
        G_yx = gaussian_kernel(srcY, srcX, beta)  # (M,B)

        A = G_xx_reg  
        B_vec = srcX   

        W = torch.linalg.solve(A, B_vec)  # (B,D)
        Y_pred = G_yx @ W   # (M,D)

        err = torch.norm(Y_pred - srcY) / torch.norm(srcY)
        print(f"Reconstruction relative error: {err.item():.4f}")

        return Y_pred

def rbf_weights(srcX, srcY, beta=1.0, lmbd=1e-6):
    """RBF weight"""
    B = srcX.shape[0]
    K = gaussian_kernel(srcX, srcX, beta)
    K_reg = K + lmbd * torch.eye(B, device=srcX.device)
    W = torch.linalg.solve(K_reg, srcY)
    return W

def rbf_predict(evalX, srcX, W, beta=1.0):
    G = gaussian_kernel(evalX, srcX, beta)  # (N,B)
    return G @ W

def deform_vertices(vertices, partitions, vertex_partition, beta=3.0):
    """
    vertices: (V,3) 
    partitions: dict, {control_X, control_Y}
    vertex_partition: (V,)
    """
    device = vertices.device
    V = vertices.shape[0]

    weights = {}
    for name, data in partitions.items():
        W = rbf_weights(data["control_X"].to(device), data["control_Y"].to(device), beta=beta)
        weights[name] = (data["control_X"].to(device), W)

    pred_displacements = torch.zeros_like(vertices)
    for name in partitions.keys():
        mask = [i for i, p in enumerate(vertex_partition) if p == name]
        if len(mask) == 0:
            continue
        mask = torch.tensor(mask, device=device, dtype=torch.long)
        verts_part = vertices[mask]
        srcX, W = weights[name]
        predY = rbf_predict(verts_part, srcX, W, beta=beta)
        pred_displacements[mask] = predY

    return vertices + pred_displacements

def area_regularization(V, F, V_orig):
    """
    V: (N,3) 
    F: (M,3) 
    V_orig: (N,3) 
    """
    vi, vj, vk = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    cross_new = torch.cross(vj - vi, vk - vi, dim=1)
    area_new = 0.5 * torch.norm(cross_new, dim=1)

    vi0, vj0, vk0 = V_orig[F[:,0]], V_orig[F[:,1]], V_orig[F[:,2]]
    cross_orig = torch.cross(vj0 - vi0, vk0 - vi0, dim=1)
    area_orig = 0.5 * torch.norm(cross_orig, dim=1)

    return ((area_new - area_orig) ** 2).sum()

def optimize_with_projection(V, F, C, lam=50.0, mu=5.0, 
                             inner_max=200, lr=1e-2, 
                             tol=1e-6, patience=10):
    """
    V: (N,3) SMPL vertices
    F: (M,3) Face 
    C: (N,3) 
    lam: Laplacian weight
    mu: aera weight
    inner_max: Iter. Number
    lr: leraning rate
    tol: convergence threshold
    patience: 
    """
    device = V.device
    V_orig = V.clone()

    L = build_cot_laplacian_torch(V_orig, F)

    V_opt = C.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([V_opt], lr=lr)

    best_loss = float("inf")
    bad_count = 0

    for it in range(inner_max):
        optimizer.zero_grad()

        lap_loss = torch.norm(torch.sparse.mm(L, (V_opt - V_orig)), p='fro')**2

        area_loss = area_regularization(V_opt, F, V_orig)

        loss = lam * lap_loss
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - tol:
            best_loss = loss.item()
            bad_count = 0
        else:
            bad_count += 1

        # if bad_count >= patience:
        #     print(f"Early stop at iter {it}, loss={loss.item():.4f}")
        #     break

        if it % 20 == 0:
            print(f"iter {it}, "
                  f"lap={lap_loss.item():.4f}, area={area_loss.item():.4f}, "
                  f"total={loss.item():.4f}")

    return V_opt.detach()



def create_dense_grid_with_knn_rotation_fixed0(A, B, smpl_verts, smpl_faces, skeleton_verts, grid_world, resolution_dhw=[64, 64, 32]):
    """
    Version that fixes the 90-degree rotation issue
    """
    Dx, Dy, Dz = resolution_dhw  # d=64, h=64, w=32
    device = A.device
    
    all_pts = torch.cat([A, B, smpl_verts], dim=0)
    grid_min, grid_max, grid_size = get_grid_parameters_corrected(all_pts, resolution_dhw)

    def coord_to_voxel_index(coords, grid_min=None, grid_max=None, resolution_dhw=None):
        if grid_min is None:
            grid_min = torch.tensor([-1, -1, -1], device=device, dtype=coords.dtype)
        if grid_max is None:
            grid_max = torch.tensor([1, 1, 1], device=device, dtype=coords.dtype)
        d,h,w = resolution_dhw
        normalized = (coords - grid_min) / (grid_max - grid_min)
        
        indices = normalized * torch.tensor([d-1, h-1, w-1], device=device, dtype=coords.dtype)
        indices = indices.unsqueeze(0).permute(0, 2, 1).reshape(1, 3, d,h,w)
        return indices

    coord_indices = coord_to_voxel_index(grid_world, resolution_dhw=resolution_dhw)
    
    d, h, w = resolution_dhw
    grid_range = grid_max - grid_min

    grid_size_corrected = torch.stack([
        grid_range[0] / d,  
        grid_range[1] / h,  
        grid_range[2] / w  
    ])
    
    print(f"Original grid_size: {grid_size}")
    print(f"Corrected grid_size: {grid_size_corrected}")
    
    occupancy = torch.zeros((Dx, Dy, Dz), dtype=torch.bool, device=device)
    id_map = torch.full((Dx, Dy, Dz), -1, dtype=torch.long, device=device)
    coord_map = torch.zeros((Dx, Dy, Dz, 3), dtype=A.dtype, device=device)

    inner_motions = smpl_verts - skeleton_verts
    smooth_vectors, smooth_with_inner_vectors, _, _ = compute_grid_knn_distance_smooth(
        grid_world, smpl_verts, inner_motions, k=1, grid_shape=resolution_dhw
    )
    print(f"Smooth vectors shape: {smooth_vectors.shape}")
    # A_center = torch.mean(A, dim=0, keepdim=True)
    
    A_sample_coords_v2 = A[:, [2, 1, 0]][:, None, None, :] 
    sampled_motions_A = F.grid_sample(
        smooth_vectors,
        A_sample_coords_v2.unsqueeze(0),  
        mode='nearest', 
        align_corners=True, 
        padding_mode="border",
    )

    sampled_motions_A = sampled_motions_A.squeeze().transpose(0, 1)

    sampled_with_inner_motions_A = F.grid_sample(
        smooth_with_inner_vectors,
        A_sample_coords_v2.unsqueeze(0),  
        mode='nearest', 
        align_corners=True, 
        padding_mode="border",
    )
    sampled_with_inner_motions_A = sampled_with_inner_motions_A.squeeze().transpose(0, 1)
    A_canonical_with_inner = A + sampled_with_inner_motions_A
    
    A_canonical = A + sampled_motions_A
    
    B_normalized = B
    B_sample_coords_v2 = B_normalized[:, [2, 1, 0]][:, None, None, :] # 正确
    
    sampled_motions_B = F.grid_sample(
        smooth_vectors,
        B_sample_coords_v2.unsqueeze(0),  
        mode='nearest', 
        align_corners=True, 
        padding_mode="border",
    )
    sampled_motions_B = sampled_motions_B.squeeze().transpose(0, 1)

    sampled_with_inner_motions_B = F.grid_sample(
        smooth_with_inner_vectors,
        B_sample_coords_v2.unsqueeze(0),  
        mode='nearest', 
        align_corners=True, 
        padding_mode="border",
    )
    sampled_with_inner_motions_B = sampled_with_inner_motions_B.squeeze().transpose(0, 1)
    B_canonical_with_inner = B + sampled_with_inner_motions_B
    
    B_canonical = B + sampled_motions_B
    k = 1
    # merge_pts = torch.cat([A_canonical_with_inner, skeleton_verts], dim=0)
    dists, knn_idx = pointutils.knn(k, A_canonical_with_inner.unsqueeze(0), B_canonical_with_inner.unsqueeze(0))
    dists[dists < 1e-6] = 1e-6
    
    knn_verts = torch.gather(
        sampled_with_inner_motions_B.expand(1, -1, -1), 1, 
        knn_idx.view(1, -1).unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
    ).view(1, A.shape[0], k, 3)
    # smooth_vectors = A.unsqueeze(2) - knn_verts
    eps = 1e-8
    if k==1:
        smooth_vectors = knn_verts.squeeze(-2) 
    else:
        weights = 1.0 / (dists + eps)  # (B, N, k)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        smooth_vectors = (weights.unsqueeze(-1) * knn_verts).sum(dim=-2)  # (B, N, 3)
    
    new_A = A_canonical_with_inner - smooth_vectors[0] 

    # new_AA = optimize_with_projection(smpl_verts, smpl_faces, new_A, lam=50.0, mu=5.0, 
    #                          inner_max=200, lr=1e-2, 
    #                          tol=1e-6, patience=10)

    # sampled_idx = (torch.randperm(smpl_verts.shape[0])[:5000]).to(smpl_verts.device)
    # new_smpl_verts = smpl_verts[sampled_idx]
    # v_skeleton = get_predictions(sigma=2.0, beta=1.0, means_= smpl_verts, srcX=B, srcY=new_A)
    visualize_point_cloud([new_A, B], [[1,0,0],[0,0,1], [0,1,0],[0.5, 0.5,0], [0.5, 0, 0.5]], faces= smpl_faces.detach().cpu().numpy(),show_coordinate_frame=False)
    return B, sampled_motions_B, B_canonical, smooth_vectors

def extract_submesh(vertices_all, faces_all, part_idx):
    """
    Args:
        vertices_all: (N, 3) numpy array, 
        faces_all:    (M, 3) numpy array, 
        part_idx:     (k,)   numpy array/list, 

    Returns:
        vertices_sub: (k, 3) numpy array,
        faces_sub:    (m, 3) numpy array, ([0,k-1])
    """
    part_idx = np.array(part_idx, dtype=np.int32)

    vertices_sub = vertices_all[part_idx]

    # Map global to local
    idx_map = {g: i for i, g in enumerate(part_idx)}

    faces_sub = []
    for tri in faces_all:
        if all(v in idx_map for v in tri):
            faces_sub.append([idx_map[v] for v in tri])

    faces_sub = np.array(faces_sub, dtype=np.int32)
    return vertices_sub, faces_sub

def create_dense_grid_with_knn_rotation(pcs, smpl_verts, skeleton_verts, smpl_faces=None, resolution_dhw=[64, 64, 32]):

    device = smpl_verts.device
    grid_world = build_grid(smpl_verts.shape[0], device, resolution_dhw=resolution_dhw)
    
    A_canonical_list, A_canonical_with_inner_list, smooth_vectors_list, smooth_with_inner_vectors_list = [], [], [], []
    inner_motions = smpl_verts - skeleton_verts
    smooth_vectors, smooth_with_inner_vectors, _, _ = compute_grid_knn_distance_smooth(
            grid_world, smpl_verts, inner_motions, k=1, grid_shape=resolution_dhw
        )
    return smooth_vectors, smooth_with_inner_vectors


def points_to_voxel_indices(points, bounds, grid_size):
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
    D, H, W = grid_size

    dx = (x_max - x_min) / D
    dy = (y_max - y_min) / H
    dz = (z_max - z_min) / W

    indices = (points - torch.tensor([x_min, y_min, z_min], device=points.device)) / \
              torch.tensor([dx, dy, dz], device=points.device)

    indices = indices.long()
    indices[:, 0].clamp_(0, D - 1)
    indices[:, 1].clamp_(0, H - 1)
    indices[:, 2].clamp_(0, W - 1)
    return indices

def accumulate_offsets(points, offsets, grid_size, bounds, flat_offsets, flat_counts):
    D, H, W = grid_size
    idx = points_to_voxel_indices(points, bounds, grid_size)
    linear_idx = idx[:,0]*H*W + idx[:,1]*W + idx[:,2]

    # flat_offsets = torch.zeros((D*H*W, 3), device=offsets.device)
    # flat_counts  = torch.zeros((D*H*W, 1), device=offsets.device)

    flat_offsets.index_add_(0, linear_idx, offsets)
    flat_counts.index_add_(0, linear_idx, torch.ones_like(flat_counts[linear_idx]))

    mask = (flat_counts.view(D, H, W) > 1e-6).float()
    flat_counts = torch.clamp(flat_counts, min=1.0)
    flat_offsets = flat_offsets / flat_counts

    grid_offsets = flat_offsets.view(D, H, W, 3)
    # mask = (flat_counts.view(D, H, W) > 1e-6).float()
    return grid_offsets, mask

def smooth_offsets(grid_offsets, mask, smoothing_kernel_size=3):
    kernel = torch.ones((1, 1, smoothing_kernel_size, smoothing_kernel_size, smoothing_kernel_size), 
                        device=grid_offsets.device)
    kernel = kernel / kernel.sum()

    # (B,C,D,H,W) 
    g = grid_offsets.permute(3,0,1,2).unsqueeze(0)  # [1,3,D,H,W]
    m = mask.unsqueeze(0).unsqueeze(0)               # [1,1,D,H,W]

    g_masked = g * m
    smoothed = []
    for c in range(3):
        conv_val = F.conv3d(g_masked[:,c:c+1], kernel, padding=smoothing_kernel_size//2)
        conv_mask = F.conv3d(m, kernel, padding=smoothing_kernel_size//2)
        smoothed_c = conv_val / (conv_mask + 1e-8)
        smoothed.append(smoothed_c)
    smoothed = torch.cat(smoothed, dim=1)  # [1,3,D,H,W]
    return smoothed #.squeeze(0).permute(1,2,3,0)

def get_dense_grid_with_smooth_vectors(pcs, offsets, resolution_dhw=[64, 64, 64], k=0, lamda_para=0.7,):
    smooth_vectors, masks = [], []
    D, H, W = resolution_dhw
    flat_offsets = torch.zeros((D*H*W, 3), device=offsets[0].device)
    flat_counts  = torch.zeros((D*H*W, 1), device=offsets[0].device)
    for i in range(len(pcs)):
        grid_offsets, mask = accumulate_offsets(pcs[i], offsets[i], grid_size=resolution_dhw, bounds=((-1,-1,-1),(1,1,1)), flat_offsets=flat_offsets, flat_counts=flat_counts)
        # smoothed_offsets = smooth_offsets(grid_offsets, mask, smoothing_kernel_size=3)
        smoothed_offsets = grid_offsets.permute(3,0,1,2).unsqueeze(0) 
        smooth_vectors.append(smoothed_offsets)
        masks.append(mask.unsqueeze(0).unsqueeze(0))

    smooth_vectors = torch.cat(smooth_vectors, dim=0)
    masks = torch.cat(masks, dim=0)
    

    for _ in range(k):
        # smooth_vectors[:, :, 1:-1, 1:-1, 1:-1] = (
        #     smooth_vectors[:, :, 2:, 1:-1, 1:-1]
        #     + smooth_vectors[:, :, :-2, 1:-1, 1:-1]
        #     + smooth_vectors[:, :, 1:-1, 2:, 1:-1]
        #     + smooth_vectors[:, :, 1:-1, :-2, 1:-1]
        #     + smooth_vectors[:, :, 1:-1, 1:-1, 2:]
        #     + smooth_vectors[:, :, 1:-1, 1:-1, :-2]
        # ) / 6.0
        mean = (
                smooth_vectors[:, :, 2:, 1:-1, 1:-1]
                + smooth_vectors[:, :, :-2, 1:-1, 1:-1]
                + smooth_vectors[:, :, 1:-1, 2:, 1:-1]
                + smooth_vectors[:, :, 1:-1, :-2, 1:-1]
                + smooth_vectors[:, :, 1:-1, 1:-1, 2:]
                + smooth_vectors[:, :, 1:-1, 1:-1, :-2]
            ) / 6.0
        # smooth_vectors[:, :, 1:-1, 1:-1, 1:-1] = (
        #         smooth_vectors[:, :, 1:-1, 1:-1, 1:-1] - mean
        #     ) * (1-lamda_para) + lamda_para * mean
        smooth_vectors[:, :, 1:-1, 1:-1, 1:-1] = (
                smooth_vectors[:, :, 1:-1, 1:-1, 1:-1] - mean
            ) * lamda_para +  mean
        # sums = smooth_vectors.sum(1, keepdim=True)
        # smooth_vectors = smooth_vectors / sums

    
    return smooth_vectors, masks

def gaussian_kernel_cross(X, Y, beta=1.0):
    """Gaussian kernel between two sets of points."""
    X_norm = (X ** 2).sum(dim=1).view(-1, 1)
    Y_norm = (Y ** 2).sum(dim=1).view(1, -1)
    dist_sq = X_norm + Y_norm - 2 * X @ Y.t()
    return torch.exp(-beta * dist_sq)

def transform_Y_from_subset(srcX_sub, srcY_sub, Y_all, Y_faces, beta=1.0, lmbd=1e-6):
    """
    Estimate transformed full set Y_all given subset correspondences.
    
    Args:
        srcX_sub: (S, D) source subset
        srcY_sub: (S, D) target subset
        Y_all: (N, D) full set of points to transform
        beta: kernel width
        lmbd: regularization
    
    Returns:
        Y_new: (N, D) transformed full set
        Delta_Y_all: (N, D) estimated motion vectors
    """
    beta=3.0
    S, D = srcX_sub.shape

    G_SS = gaussian_kernel_cross(Y_all, Y_all, beta)  # (S, S)
    G_SS_reg = G_SS + lmbd * torch.eye(Y_all.shape[0], device=Y_all.device)
    
    W = torch.linalg.solve(G_SS_reg, Y_all)  # (S, D)
    G_YS = gaussian_kernel_cross(srcX_sub, Y_all, beta)  # (N, S)

 
    Y_new = G_YS @ W  # (N, D)

    # correspondence_idx = G_YS.argmax(dim=1)  # (N,)
    # corresponding_Y = Y_all[correspondence_idx]  # (N, D)
    weights = G_YS / (G_YS.sum(dim=1, keepdim=True) + 1e-8)  # 
    correspondence_idx = weights.argmax(dim=1)
    corresponding_Y = Y_all[correspondence_idx]

    k = 5
    # merge_pts = torch.cat([A_canonical_with_inner, skeleton_verts], dim=0)
    dists, knn_idx = pointutils.knn(k, Y_all.unsqueeze(0), corresponding_Y.unsqueeze(0))
    dists[dists < 1e-6] = 1e-6
    
    knn_verts = torch.gather(
        Y_new.expand(1, -1, -1), 1, 
        knn_idx.view(1, -1).unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
    ).view(1, Y_all.shape[0], k, 3)
    knn_verts = Y_all.unsqueeze(0).unsqueeze(2) - knn_verts
    eps = 1e-8
    if k==1:
        smooth_vectors = knn_verts.squeeze(-2) 
    else:
        weights = 1.0 / (dists + eps)  # (B, N, k)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        smooth_vectors = (weights.unsqueeze(-1) * knn_verts).sum(dim=-2)  # (B, N, 3)
    
    new_A = Y_all - smooth_vectors[0] 
    visualize_point_cloud([Y_new, srcX_sub,], [[1,0,0],[0,0,1], [0,1,0],[0.5, 0.5,0], [0.5, 0, 0.5]], Y_faces.detach().cpu().numpy(), show_coordinate_frame=False)
    
    return Y_new


def create_dense_grid_with_knn_rotation_fixed(A, B, smpl_verts, smpl_faces, skeleton_verts, grid_world, resolution_dhw=[64, 64, 32]):
    Dx, Dy, Dz = resolution_dhw  # d=64, h=64, w=32
    device = A.device

    all_pts = torch.cat([A, B, smpl_verts], dim=0)
    grid_min, grid_max, grid_size = get_grid_parameters_corrected(all_pts, resolution_dhw)

    def coord_to_voxel_index(coords, grid_min=None, grid_max=None, resolution_dhw=None):
        if grid_min is None:
            grid_min = torch.tensor([-1, -1, -1], device=device, dtype=coords.dtype)
        if grid_max is None:
            grid_max = torch.tensor([1, 1, 1], device=device, dtype=coords.dtype)
        d,h,w = resolution_dhw
        normalized = (coords - grid_min) / (grid_max - grid_min)
        
        indices = normalized * torch.tensor([d-1, h-1, w-1], device=device, dtype=coords.dtype)
        indices = indices.unsqueeze(0).permute(0, 2, 1).reshape(1, 3, d,h,w)
        return indices

    coord_indices = coord_to_voxel_index(grid_world, resolution_dhw=resolution_dhw)
    
    d, h, w = resolution_dhw
    grid_range = grid_max - grid_min
    
    grid_size_corrected = torch.stack([
        grid_range[0] / d,  
        grid_range[1] / h,  
        grid_range[2] / w   
    ])
    
    print(f"Original grid_size: {grid_size}")
    print(f"Corrected grid_size: {grid_size_corrected}")
    
    occupancy = torch.zeros((Dx, Dy, Dz), dtype=torch.bool, device=device)
    id_map = torch.full((Dx, Dy, Dz), -1, dtype=torch.long, device=device)
    coord_map = torch.zeros((Dx, Dy, Dz, 3), dtype=A.dtype, device=device)

    # occupancy2 = torch.zeros((Dx, Dy, Dz), dtype=torch.bool, device=device)
    # id_map2 = torch.full((Dx, Dy, Dz), -1, dtype=torch.long, device=device)
    # coord_map2 = torch.zeros((Dx, Dy, Dz, 3), dtype=A.dtype, device=device)

    inner_motions = smpl_verts - skeleton_verts
    smooth_vectors, smooth_with_inner_vectors, _, _ = compute_grid_knn_distance_smooth(
        grid_world, smpl_verts, inner_motions, k=1, grid_shape=resolution_dhw
    )
    print(f"Smooth vectors shape: {smooth_vectors.shape}")
    # A_center = torch.mean(A, dim=0, keepdim=True)
    
    A_sample_coords_v2 = A[:, [2, 1, 0]][:, None, None, :] 
    sampled_motions_A = F.grid_sample(
        smooth_vectors,
        A_sample_coords_v2.unsqueeze(0),  
        mode='nearest', 
        align_corners=True, 
        padding_mode="border",
    )

    sampled_motions_A = sampled_motions_A.squeeze().transpose(0, 1)

    sampled_with_inner_motions_A = F.grid_sample(
        smooth_with_inner_vectors,
        A_sample_coords_v2.unsqueeze(0),  
        mode='nearest', 
        align_corners=True, 
        padding_mode="border",
    )
    sampled_with_inner_motions_A = sampled_with_inner_motions_A.squeeze().transpose(0, 1)
    A_canonical_with_inner = A + sampled_with_inner_motions_A
    
    # A_canonical = A + sampled_motions_A
    A = smpl_verts
    A_sample_coords_v2 = A[:, [2, 1, 0]][:, None, None, :] # 正确
    idx_A = F.grid_sample(
        coord_indices,
        A_sample_coords_v2.unsqueeze(0),  
        mode='nearest', 
        align_corners=True, 
        padding_mode="border",
    )
    idx_A = idx_A.squeeze().transpose(0, 1).long()
    # idx_A = world_to_grid_coords(A, grid_min, grid_size_corrected)
    valid_mask_A = ((idx_A >= 0) & (idx_A < torch.tensor([Dx, Dy, Dz], device=device))).all(dim=1)
    idx_A = idx_A[valid_mask_A]
    A_valid = A[valid_mask_A]
    ids = torch.arange(A_valid.shape[0], device=device)
    
    occupancy[idx_A[:, 0], idx_A[:, 1], idx_A[:, 2]] = True
    id_map[idx_A[:, 0], idx_A[:, 1], idx_A[:, 2]] = ids
    coord_map[idx_A[:, 0], idx_A[:, 1], idx_A[:, 2]] = A_valid

    # grid_world_c = grid_world.reshape(Dx, Dy, Dz, 3)
    # a =  grid_world_c[idx_A[:, 0], idx_A[:, 1], idx_A[:, 2], :]

    
    B_normalized = B
    B_sample_coords_v2 = B_normalized[:, [2, 1, 0]][:, None, None, :] # 正确
    
    sampled_motions_B = F.grid_sample(
        smooth_vectors,
        B_sample_coords_v2.unsqueeze(0),  
        mode='nearest', 
        align_corners=True, 
        padding_mode="border",
    )
    sampled_motions_B = sampled_motions_B.squeeze().transpose(0, 1)

    sampled_with_inner_motions_B = F.grid_sample(
        smooth_with_inner_vectors,
        B_sample_coords_v2.unsqueeze(0),  
        mode='nearest', 
        align_corners=True, 
        padding_mode="border",
    )
    sampled_with_inner_motions_B = sampled_with_inner_motions_B.squeeze().transpose(0, 1)
    B_canonical_with_inner = B + sampled_with_inner_motions_B
    
    B_canonical = B + sampled_motions_B
    B_sample_coords_v2 = B_canonical_with_inner[:, [2, 1, 0]][:, None, None, :] # 正确
    idx_B = F.grid_sample(
        coord_indices,
        B_sample_coords_v2.unsqueeze(0),  
        mode='nearest', 
        align_corners=True, 
        padding_mode="border",
    )
    idx_B = idx_B.squeeze().transpose(0, 1).long()
    valid_mask_B = ((idx_B >= 0) & (idx_B < torch.tensor([Dx, Dy, Dz], device=device))).all(dim=1)
    valid_mask = torch.logical_and(valid_mask_B, occupancy[idx_B[:,0], idx_B[:,1], idx_B[:,2]].view(-1))
    # idx_B = idx_B[valid_mask_B]
    # B_valid = B[valid_mask_B]
    # ids = torch.arange(B_valid.shape[0], device=device)

    # occupancy2[idx_B[:, 0], idx_B[:, 1], idx_B[:, 2]] = True
    # id_map2[idx_B[:, 0], idx_B[:, 1], idx_B[:, 2]] = ids
    # coord_map2[idx_B[:, 0], idx_B[:, 1], idx_B[:, 2]] = B_valid

    known_pts = coord_map[idx_B[:,0][valid_mask], idx_B[:,1][valid_mask], idx_B[:,2][valid_mask]]
    motion_A = known_pts - B[valid_mask]

    # transform_Y_from_subset(B, B_canonical, smpl_verts, 1.0)

    k = 1
    # merge_pts = torch.cat([A_canonical_with_inner, skeleton_verts], dim=0)
    dists, knn_idx = pointutils.knn(k, smpl_verts.unsqueeze(0), B_canonical.unsqueeze(0))
    dists[dists < 1e-6] = 1e-6
    
    knn_verts = torch.gather(
        B.expand(1, -1, -1), 1, 
        knn_idx.view(1, -1).unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
    ).view(1, A.shape[0], k, 3)
    knn_verts = smpl_verts.unsqueeze(0).unsqueeze(2) - knn_verts
    eps = 1e-8
    if k==1:
        smooth_vectors = knn_verts.squeeze(-2) 
    else:
        weights = 1.0 / (dists + eps)  # (B, N, k)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        smooth_vectors = (weights.unsqueeze(-1) * knn_verts).sum(dim=-2)  # (B, N, 3)
    
    new_A = smpl_verts - smooth_vectors[0] 

    # v_skeleton = get_predictions(sigma=2.0, beta=1.0, means_= smpl_verts, srcX=B, srcY=new_A)

    # known_pts = coord_map2[idx_B[:,0], idx_B[:,1], idx_B[:,2]]
    # unknown_pts = coord_map[idx_A[:,0], idx_A[:,1], idx_A[:,2]]

    # k = 1
    # # merge_pts = torch.cat([A_canonical_with_inner, skeleton_verts], dim=0)
    # dists, knn_idx = pointutils.knn(k, skeleton_verts.unsqueeze(0), B_canonical_with_inner.unsqueeze(0))
    # dists[dists < 1e-6] = 1e-6
    
    # knn_verts = torch.gather(
    #     sampled_with_inner_motions_B.expand(1, -1, -1), 1, 
    #     knn_idx.view(1, -1).unsqueeze(-1).expand(-1, -1, 3).to(torch.int64)
    # ).view(1, A.shape[0], k, 3)
    # # smooth_vectors = A.unsqueeze(2) - knn_verts
    # eps = 1e-8
    # if k==1:
    #     smooth_vectors = knn_verts.squeeze(-2) 
    # else:
    #     weights = 1.0 / (dists + eps)  # (B, N, k)
    #     weights = weights / weights.sum(dim=-1, keepdim=True)
    #     smooth_vectors = (weights.unsqueeze(-1) * knn_verts).sum(dim=-2)  # (B, N, 3)
    
    # new_A = skeleton_verts - smooth_vectors[0] 

    new_AA = optimize_with_projection(smpl_verts, smpl_faces, new_A, lam=50.0, mu=5.0, 
                             inner_max=2000, lr=1e-2, 
                             tol=1e-6, patience=10)
    # # sampled_idx = (torch.randperm(smpl_verts.shape[0])[:5000]).to(smpl_verts.device)
    # # new_smpl_verts = smpl_verts[sampled_idx]
    # v_skeleton = get_predictions(sigma=2.0, beta=1.0, means_= smpl_verts, srcX=B, srcY=new_A)
    # # # smooth_vectors2 = new_A - A
    # # new_skeleton_verts = skeleton_verts - v_skeleton
    # # visualize_point_cloud([B_canonical_with_inner, skeleton_verts, A_canonical_with_inner], [[1,0,0],[0,0,1], [0,1,0],[0.5, 0.5,0], [0.5, 0, 0.5]], show_coordinate_frame=False)
    visualize_point_cloud([new_AA, B], [[1,0,0],[0,0,1], [0,1,0],[0.5, 0.5,0], [0.5, 0, 0.5]], show_coordinate_frame=False)
    return B, sampled_motions_B, B_canonical, smooth_vectors


def create_dense_grid_with_knn(A, B, smpl_verts, grid_world, grid_origin, grid_size, grid_shape):
    Dx, Dy, Dz = grid_shape
    device = A.device
    
    occupancy = torch.zeros((Dx, Dy, Dz), dtype=torch.bool, device=device)
    id_map = torch.full((Dx, Dy, Dz), -1, dtype=torch.long, device=device)
    coord_map = torch.zeros((Dx, Dy, Dz, 3), dtype=A.dtype, device=device)
    
    idx_A = ((A) / grid_size).long()
    valid_mask_A = ((idx_A >= 0) & (idx_A < torch.tensor([Dx, Dy, Dz], device=device))).all(dim=1)
    idx_A = idx_A[valid_mask_A]
    A_valid = A[valid_mask_A]
    ids = torch.arange(A_valid.shape[0], device=device)
    
    occupancy[idx_A[:, 0], idx_A[:, 1], idx_A[:, 2]] = True
    id_map[idx_A[:, 0], idx_A[:, 1], idx_A[:, 2]] = ids
    coord_map[idx_A[:, 0], idx_A[:, 1], idx_A[:, 2]] = A_valid

    smooth_vectors, _, _ = compute_grid_knn_distance_smooth(
        grid_world, smpl_verts, k=1, grid_shape=grid_shape
    )
    
    grid_extent = torch.tensor([Dx, Dy, Dz], device=device, dtype=B.dtype) * grid_size
    grid_max = grid_origin + grid_extent
    
    B_normalized = 2.0 * (B - grid_origin) / (grid_max - grid_origin) - 1.0
    B_normalized = B_normalized.clamp(-1, 1)  
    
    B_sample_coords = B_normalized[:, None, None, :]  # [N, 1, 1, 3]
    
    sampled_motions_B = F.grid_sample(
        smooth_vectors.unsqueeze(0),  # [1, 3, Dx, Dy, Dz]
        B_sample_coords.unsqueeze(0),  # [1, N, 1, 1, 3]
        mode='bilinear', 
        align_corners=True, 
        padding_mode="border",
    )  # [1, 3, N, 1, 1]
    
    sampled_motions_B = sampled_motions_B.squeeze().transpose(0, 1)  # [N, 3]
    
    B_canical = B + sampled_motions_B
    idx_B = B_canical[0] #[N,3]
    valid_mask_B = ((idx_B >= 0) & (idx_B < torch.tensor([Dx, Dy, Dz], device=device))).all(dim=1)
    valid_mask = torch.logical_and(valid_mask_B, occupancy[idx_B[:,0], idx_B[:,1], idx_B[:,2]].view(-1))

    motion_A = coord_map[idx_B[:,0], idx_B[:,1], idx_B[:,2]] - grid_B
    motion_A = motion_A[valid_mask] 

    known_idx = idx_B          # (N_known,3)
    known_motion = motion_A    # (N_known,3)

    grid_flat = torch.stack(torch.meshgrid(
        torch.arange(Dx,device=device),
        torch.arange(Dy,device=device),
        torch.arange(Dz,device=device)
    ),-1).reshape(-1,3).float()   # (N_voxel,3)

    # knn_points(query, support)
    dists, idx, _ = knn_points(
        grid_flat[None],       # [1, N_voxel, 3]
        known_idx[None].float(), # [1, N_known, 3]
        K=1
    )

    nn_idx = idx[0,:,0]  # (N_voxel,)
    motion_full = known_motion[nn_idx]  # (N_voxel,3)
    motion_full = motion_full.view(Dx,Dy,Dz,3)

    min_dists = dists[0, :, 0].view(Dx, Dy, Dz)
    smpl_idx_grid = smpl_nn_idx[0, :, 0].view(Dx, Dy, Dz) 
    
    return occupancy, id_map, coord_map, min_dists, smpl_idx_grid, smooth_vectors, grid_world

def get_on_mesh_init_geo_values(v, f, opacity_init_logit, on_mesh_subdivide = 0, scale_init_factor = 1.0, thickness_init_factor = 0.5, max_scale = 1.0, min_scale = 0.0, s_inv_act = torch.logit):
        x_all, q_all, s_all, o_all = [], [], [], []
        for i in range(len(v)):
            x, mesh = init_xyz_on_mesh(v[i], f, on_mesh_subdivide)
            q, s, o = init_qso_on_mesh(
                mesh,
                scale_init_factor,
                thickness_init_factor,
                max_scale,
                min_scale,
                s_inv_act,
                opacity_init_logit,
            )
            
            x_all.append(x)
            q_all.append(q)
            s_all.append(s)
            o_all.append(o)
        
        x_all = torch.cat(x_all, dim=0)
        q_all = torch.cat(q_all, dim=0)
        s_all = torch.cat(s_all, dim=0)
        o_all = torch.cat(o_all, dim=0)
        return x_all, q_all, s_all, o_all

    
def get_vertices_mapping_to_skeleton(vertices_template, weights, J):  
    vertices_proj = weights @ J  # (6890, 3)

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(vertices_template[:, 0], vertices_template[:, 1], vertices_template[:, 2],
                s=0.5, color='blue')
    ax1.scatter(J[:, 0], J[:, 1], J[:, 2], s=30, color='red')  # 关节点
    ax1.set_title("Original Mesh + Joints")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(vertices_proj[:, 0], vertices_proj[:, 1], vertices_proj[:, 2],
                s=0.5, color='green')
    ax2.scatter(J[:, 0], J[:, 1], J[:, 2], s=30, color='red')
    ax2.set_title("Vertices Projected to Skeleton")

    plt.tight_layout()
    plt.show()

def normalize_vector(v, eps=1e-8):
    """L2 normalize along last dim"""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return v / norm

def compute_rot_from_quaternion_and_align(v, masked_theta, x):
    """
    v: (B, 3)
    masked_theta: (num_instances, 4) quaternion 
    x:
    """
    q = masked_theta[:, 0] / masked_theta[:, 0].norm(dim=-1, keepdim=True)  


    rot_per_pts = quat_to_rotmat(q)  # (num_instances, 3, 3)

    v = torch.matmul(v.unsqueeze(1), rot_per_pts.transpose(1, 2))  # (B, 1, 3)
    v = v.squeeze(1)  # (B, 3)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

    z_axis = torch.tensor([0., 0., 1.], device=x.device, dtype=x.dtype)[None]  # (1,3)
    z_axis = z_axis.expand_as(v)

    axis = torch.cross(v, z_axis, dim=1)  # (B,3)
    axis_norm = axis.norm(dim=1)          # (B,)
    mask_zero = axis_norm < 1e-8

    dot = (v * z_axis).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)               # (B,)
    rotvec = torch.zeros_like(v)
    rotvec[~mask_zero] = axis[~mask_zero] / axis_norm[~mask_zero, None] * angle[~mask_zero, None]

    rot_mat = axis_angle_to_matrix(rotvec)  # (B,3,3)

    if mask_zero.any():
        rot_mat[mask_zero] = torch.eye(3, device=x.device, dtype=x.dtype)

    return rot_mat, rot_per_pts

def get_smpl_template(num_human, init_beta, instances_quats, instances_trans, cano_pose_type="da_pose", smpl_points_num=6890, tp_parents=None, joints24=None, J_canonical=None, A0=None, W_init=None, v_init=None, faces=None, x=None, smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl", use_canical=True, use_correction=True):
    assert num_human == init_beta.shape[0], "num_human should be the same as the number of beta"
    is_cpu = True
    if init_beta.device.type == "cpu":
        is_cpu = True
        init_beta = torch.as_tensor(init_beta, dtype=torch.float32).cpu()
        instances_quats = torch.as_tensor(instances_quats, dtype=torch.float32).cpu()
        instances_trans = torch.as_tensor(instances_trans, dtype=torch.float32).cpu()
    else:
        is_cpu = False
    

    if tp_parents is None:
        cano_pose_type = cano_pose_type
        can_pose = get_predefined_human_rest_pose(cano_pose_type)
        can_pose = axis_angle_to_matrix(torch.cat([torch.zeros(1, 3), can_pose], 0))
        _template_layer = SMPLLayer(model_path=smpl_model_path)
        init_smpl_output, joints24 = _template_layer(
            betas=init_beta,
            body_pose=can_pose[None, 1:].repeat(num_human, 1, 1, 1),
            global_orient=can_pose[None, 0].repeat(num_human, 1, 1, 1),
            return_full_pose=True,
        )
        J_canonical, A0 = init_smpl_output.J, init_smpl_output.A
        A0_inv = torch.inverse(A0)

        v_init = init_smpl_output.vertices  # [B, 6890, 3]
        W_init = _template_layer.lbs_weights  # [6890, 24]
        faces = _template_layer.faces_tensor
        # instances_quats[1:,1:] = torch.zeros_like(instances_quats[1:,1:])
        opacity_init_value = 0.99
        x, q, s, o = get_on_mesh_init_geo_values(v_init, faces, opacity_init_logit=torch.logit(torch.tensor(opacity_init_value)))
    else:
        A0_inv = torch.inverse(A0)

    if instances_quats.ndim == 2:
        masked_theta = instances_quats.unsqueeze(0)  # [23,24,4]->[11,24,4]
    else:
        masked_theta = instances_quats  # [23,24,4]->[11,24,4]
    masked_theta = masked_theta / masked_theta.norm(dim=-1, keepdim=True)

    assert (
        masked_theta.ndim == 3 and masked_theta.shape[-1] == 4
    ), "pose should have shape Bx24x3, in axis-angle format"
    # nB = len(masked_theta)
    if tp_parents is None:
        _, A = batch_rigid_transform(quaternion_to_matrix(masked_theta), J_canonical, _template_layer.parents)
    else:
        _, A = batch_rigid_transform(quaternion_to_matrix(masked_theta), J_canonical, tp_parents)
    A = torch.einsum("bnij, bnjk->bnik", A, A0_inv)  # B,24,4,4
    W = W_init.unsqueeze(0).repeat(num_human, 1, 1)

    T = torch.einsum("bnj, bjrc -> bnrc", W, A)
    R = T[:, :, :3, :3] # [N, 3, 3]
    t = T[:, :, :3, 3]  # [N, 3]


    ###########################
    rot_mat, rot_per_pts = None, None
    if use_correction:
        pelvis_coord = J_canonical[:, 0, :]   
        neck_coord   = J_canonical[:, 12, :]  
        v = neck_coord - pelvis_coord
        rot_mat, rot_per_pts = compute_rot_from_quaternion_and_align(v, masked_theta, x)
    ########################
    
    if use_canical:
        if use_correction:
            deformed_means = torch.bmm(x.reshape(num_human,6890,3), rot_per_pts.transpose(1,2))
            undeformed_means = deformed_means.clone()
            deformed_means = torch.bmm(deformed_means, rot_mat.transpose(1,2))
            x_joints24 = torch.matmul(joints24, rot_per_pts.transpose(1,2))  # (B, J, 3)
            x_joints24 = torch.matmul(x_joints24, rot_mat.transpose(1,2))             # (B, J, 3)
            deformed_means = deformed_means.reshape(num_human, -1, 3).cpu().to(x.dtype)
        else:
            deformed_means = x.reshape(num_human, -1, 3).cpu().to(x.dtype)
        vertices_proj = torch.bmm(W, x_joints24)     
    else:
        reshaped_means = x.reshape(num_human, smpl_points_num, 3)
        can_rot = R[0,0,:,:]
        deformed_means = torch.einsum("bnij,bnj->bni", R, reshaped_means) + t # [N, 6890, 3]
        undeformed_means = deformed_means.clone()
        if use_correction:
            deformed_means = torch.bmm(deformed_means, rot_mat.transpose(1,2))
        if is_cpu:
            deformed_means   = deformed_means.reshape(num_human, -1, 3).cpu().to(x.dtype)
            undeformed_means = undeformed_means.reshape(num_human, -1, 3).cpu().to(x.dtype)
        else:
            deformed_means   = deformed_means.reshape(num_human, -1, 3).to(x.dtype)
            undeformed_means = undeformed_means.reshape(num_human, -1, 3).to(x.dtype)

        if use_correction:
            joints24 = torch.matmul(joints24, can_rot.unsqueeze(0).transpose(1,2))  # (B, J, 3)
            joints24 = torch.matmul(joints24, rot_mat.transpose(1,2))             # (B, J, 3)

        vertices_proj = torch.bmm(W, joints24)    

    bbox_min = deformed_means.min(dim=1)[0]
    bbox_max = deformed_means.max(dim=1)[0]
    local_shift = (bbox_min + bbox_max) / 2
    instances_trans = instances_trans - local_shift
    # visualize_point_cloud([deformed_means[0]], [[0,0,1]])
    deformed_means_o2w = deformed_means + instances_trans.unsqueeze(1)
    undeformed_means_o2w = undeformed_means + instances_trans.unsqueeze(1)
    vertices_proj_o2w = vertices_proj + instances_trans.unsqueeze(1)
    return deformed_means_o2w.contiguous(), _, faces, vertices_proj_o2w.to(dtype=deformed_means_o2w.dtype).contiguous(), undeformed_means_o2w

def quat_act(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)

@torch.no_grad()
def batch_kabsch_weighted(A, B, w=None, eps=1e-8):
    """
    A,B: (B,N,K,3)
    w:   (B,N,K,1) or None
    return:
      R: (B,N,3,3)  in SO(3)
      S: (B,N,3)    singular values of H
    """
    if w is None:
        H = torch.einsum('bnki,bnkj->bnij', A, B)
    else:
        H = torch.einsum('bnk,bnki,bnkj->bnij', w.squeeze(-1), A, B)

    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)

    R0 = V @ U.transpose(-2, -1)
    det = torch.linalg.det(R0)

    D = torch.eye(3, device=A.device, dtype=A.dtype).view(1,1,3,3).expand(A.shape[0], A.shape[1], 3, 3).clone()
    D[..., 2, 2] = torch.where(det < 0, -1.0, 1.0)

    R = V @ D @ U.transpose(-2, -1)
    return R, S

def compute_node_rotations(P1, P2, knn_idx, sigma_scale=1.0, bad_ratio=1e-3, R_prev=None):
    """
    P1,P2: (B,N,3)
    knn_idx: (B,N,K)  
    R_prev: (B,N,3,3) 
    """
    # neighbors: (B,N,K,3)
    P1_neigh = pointutils.grouping_operation(P1.transpose(1,2).contiguous(), knn_idx).permute(0,2,3,1).contiguous()
    P2_neigh = pointutils.grouping_operation(P2.transpose(1,2).contiguous(), knn_idx).permute(0,2,3,1).contiguous()

    # edge vectors (centered at the vertex itself)
    P1_mean = P1_neigh.mean(dim=2, keepdim=True) #(B,N,1,3)
    P2_mean = P2_neigh.mean(dim=2, keepdim=True) #(B,N,1,3)
    A = P1_neigh - P1_mean
    B = P2_neigh - P2_mean

    # # adaptive gaussian weights from P1 distances
    d2 = (A**2).sum(dim=-1, keepdim=True)                            # (B,N,K,1)
    dist = torch.sqrt(d2 + 1e-12)
    sigma = dist.median(dim=2, keepdim=True).values * sigma_scale + 1e-6
    w = torch.exp(-d2 / (sigma*sigma + 1e-8))

    R, S = batch_kabsch_weighted(A, B, w=w)

    # degeneracy detection: S3/S1 small => unstable
    ratio = S[..., 2] / (S[..., 0] + 1e-12)                          # (B,N)
    bad = ratio < bad_ratio

    if R_prev is not None:
        R = torch.where(bad[..., None, None], R_prev, R)
    else:
        I = torch.eye(3, device=R.device, dtype=R.dtype).view(1,1,3,3)
        R = torch.where(bad[..., None, None], I, R)

    return R, ratio

def deform_with_graph_batch(X1, G1, G2, R, knn_idx, knn_dist):
    """
    Args:
        X1: (M,3) 
        G1: (B,N,3) 
        G2: (B,N,3) 
        R:  (B,N,3,3)
        knn_idx: (B,M,K) 
        knn_dist: (B,M,K) 
    Returns:
        X2: (M,3) 
    """
    sigma2 = knn_dist.mean()
    weights = torch.exp(-knn_dist**2 / (2*sigma2))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # (M,K)

    G1_neigh = pointutils.grouping_operation(G1.transpose(1,2).contiguous(), knn_idx)  # (B,3,N,K)
    G1_neigh = G1_neigh[0].permute(1,2,0).contiguous()
    G2_neigh = pointutils.grouping_operation(G2.transpose(1,2).contiguous(), knn_idx)  # (B,3,N,K)
    G2_neigh = G2_neigh[0].permute(1,2,0).contiguous()

    B, N, _, _ = R.shape
    R = R.reshape(B,N,-1).contiguous()
    R_neigh = pointutils.grouping_operation(R.transpose(1,2).contiguous(), knn_idx)  # (B,9,M,K)
    R_neigh = R_neigh[0].permute(1,2,0).reshape(R_neigh.shape[2],R_neigh.shape[3],3,3).contiguous() # (M,K,3,3)

    d = X1.unsqueeze(1) - G1_neigh   # (M,K,3)
    d_rot = torch.einsum('mkij,mkj->mki', R_neigh, d)  # (M,K,3)

    contrib = d_rot + G2_neigh       # (M,K,3)
    X2 = (weights.unsqueeze(-1) * contrib).sum(dim=1) # (M,3)
    delta_G2 = (weights.unsqueeze(-1) * d_rot).sum(dim=1) # (M,3)
    return X2, G2_neigh.mean(dim=1)
  
# --------------------------
# Pipeline
# --------------------------
def deformation_graph_pipeline(P1, P2, X1, quats_P1=None, quats_P2=None, X_quats=None, S1=None, n_nodes=500, K_node=16, K_point=3):
    """
    deformation graph pipeline
    Args:
        P1: (N,3) SMPL Pose1
        P2: (N,3) SMPL Pose2
        X1: (M,3) Point cloud Pose1
        n_nodes:  FPS
        K_node: 
        K_point: 
    Returns:
        X2: (M,3) Point cloud  Pose2
    """
    node_quats_P1=None
    node_quats_P2=None
    P1 = P1.unsqueeze(0).contiguous()
    P2 = P2.unsqueeze(0).contiguous()
    X1 = X1.unsqueeze(0).contiguous()

    node_idx = pointutils.furthest_point_sample(P1, n_nodes)   # (G,)
    # G1, G2 = P1[node_idx], P2[node_idx]  # (G,3)
    G1 = pointutils.gather_operation(P1.transpose(1,2).contiguous(), node_idx)
    G2 = pointutils.gather_operation(P2.transpose(1,2).contiguous(), node_idx)
    G1 = G1.permute(0,2,1).contiguous()
    G2 = G2.permute(0,2,1).contiguous()
    if quats_P1 is not None and quats_P2 is not None:
        node_quats_P1 = pointutils.gather_operation(quats_P1.transpose(1,2).contiguous(), node_idx)
        node_quats_P2 = pointutils.gather_operation(quats_P2.transpose(1,2).contiguous(), node_idx)
        node_quats_P1 = node_quats_P1.permute(0,2,1).contiguous()
        node_quats_P2 = node_quats_P2.permute(0,2,1).contiguous()

   

    if node_quats_P1 is not None and node_quats_P2 is not None:
        Qr = quaternion_multiply(node_quats_P2, quaternion_invert(node_quats_P1))  # (1,G,4)
        R = quaternion_to_matrix(Qr)  # (1,G,3,3)
    else:
        _, knn_idx_nodes = pointutils.knn(K_node, G1, P1)
        R = compute_node_rotations(P1, P2, knn_idx_nodes)  # (G,3,3)
        R = R.unsqueeze(0)
        Qr = matrix_to_quaternion(R.reshape(-1,3,3)).reshape(1,n_nodes,4)


    knn_dist, knn_idx_points = pointutils.knn(K_point, X1, G1)  # (1,M,K_point), (1,M,K_point)

    if X_quats is not None and S1 is not None:
        # Gaussian: mean+quat+scale 
        X2, Q2, S2, G1_neigh = deform_with_graph_batch_gaussian_full(
            X1[0], X_quats, S1, G1, G2, R, Qr, knn_idx_points, knn_dist[0]
        )
        return X2, Q2, S2
    elif X_quats is not None:
        # Only mean + quaternion
        X2, Q2, _, G1_neigh = deform_with_graph_batch_gaussian_full(
            X1[0], X_quats, torch.ones_like(X1[0]), G1, G2, R, Qr, knn_idx_points, knn_dist[0]
        )
        return X2, Q2
    else:
        # mean
        X2 = deform_with_graph_batch(X1[0], G1, G2, R, knn_idx_points, knn_dist[0])
        return X2


def deformation_graph_pipeline_fast(G1, G2, R, Qr, X1, Trans=None, P1=None, P2=None, X1_quats=None, S1=None, K_point=3, smpl_mode=False):
    """
    完整 deformation graph pipeline
    Args:
        P1: (N,3) SMPL Pose1
        P2: (N,3) SMPL Pose2
        X1: (M,3) Point cloud Pose1
        n_nodes: the number of FPS points
        K_node: For R
        K_point: 
    Returns:
        X2: (M,3) Point cloud Pose2
    """
    G1 = G1.unsqueeze(0).contiguous()
    G2 = G2.unsqueeze(0).contiguous()
    R = R.unsqueeze(0).contiguous()
    Qr = Qr.unsqueeze(0).contiguous()
    X1 = X1.unsqueeze(0).contiguous()
    if Trans is not None:
        Trans =  Trans.unsqueeze(0).contiguous()

    knn_dist, knn_idx_points = pointutils.knn(K_point, X1, G1)  # (1,M,K_point), (1,M,K_point)

    if X1_quats is not None and S1 is not None:
        # Gaussian: mean+quat+scale 
        X2, Q2, S2 = deform_with_graph_batch_gaussian_full(
            X1[0], X1_quats, S1, G1, G2, R, Qr, Trans, P2, knn_idx_points, knn_dist[0], smpl_mode)
        return X2, Q2, S2
    elif X1_quats is not None:
        # mean + quaternion
        X2, Q2, _ = deform_with_graph_batch_gaussian_full(
            X1[0], X1_quats, None, G1, G2, R, Qr, Trans, P2, knn_idx_points, knn_dist[0], smpl_mode)
        return X2, Q2, None
    else:
        # mean
        X2, _ = deform_with_graph_batch(X1[0], G1, G2, R, knn_idx_points, knn_dist[0])
        return X2, None, None
    
def quaternion_multiply(q1, q2):
    """
    Hamilton product of two quaternions.
    q1, q2: (...,4) [w,x,y,z]
    """
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w,x,y,z),dim=-1)

def deform_with_graph_batch_gaussian(X1, Q1, G1, G2, R, Qr, knn_idx, knn_dist):
    """
    Args:
        X1: (M,3) Gauss means Pose1
        Q1: (M,4) Gauss orientations Pose1 (unit quats)
        G1: (B,N,3) graph nodes Pose1
        G2: (B,N,3) graph nodes Pose2
        R:  (B,N,3,3) node rotation matrix
        Qr: (B,N,4) node rotations quaternion
        knn_idx: (B,M,K)
        knn_dist: (B,M,K)
    Returns:
        X2: (M,3) transformed means
        Q2: (M,4) transformed quaternions
    """
    sigma2 = knn_dist.mean()
    weights = torch.exp(-knn_dist**2 / (2*sigma2))
    weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)  # (B,M,K)

    ## gather node positions
    G1_neigh = pointutils.grouping_operation(G1.transpose(1,2).contiguous(), knn_idx)[0].permute(1,2,0)  # (M,K,3)
    G2_neigh = pointutils.grouping_operation(G2.transpose(1,2).contiguous(), knn_idx)[0].permute(1,2,0)  # (M,K,3)

    ## gather node rotations
    B, N, _, _ = R.shape
    Rv = R.reshape(B,N,-1)
    R_neigh = pointutils.grouping_operation(Rv.transpose(1,2).contiguous(), knn_idx)[0].permute(1,2,0)
    R_neigh = R_neigh.reshape(R_neigh.shape[0],R_neigh.shape[1],3,3)  # (M,K,3,3)

    Qr_neigh = pointutils.grouping_operation(Qr.transpose(1,2).contiguous(), knn_idx)  # (B,4,N,K)
    Qr_neigh = Qr_neigh[0].permute(1,2,0)  # (M,K,4)

    ## means deformation
    d = X1.unsqueeze(1) - G1_neigh   # (M,K,3)
    d_rot = torch.einsum('mkij,mkj->mki', R_neigh, d)  # (M,K,3)
    contrib = d_rot + G2_neigh       # (M,K,3)
    X2 = (weights.unsqueeze(-1) * contrib).sum(dim=1)  # (M,3)

    ## quaternion deformation
    Q1_expand = Q1.unsqueeze(1).expand_as(Qr_neigh)  # (M,K,4)
    Qj = quaternion_multiply(Qr_neigh, Q1_expand)    # apply node rotation
    Q_weighted = (weights.unsqueeze(-1) * Qj).sum(dim=1)  # (M,4)
    Q2 = F.normalize(Q_weighted, dim=-1)

    return X2, Q2

def deform_with_graph_batch_gaussian_full(X1, Q1, S1, G1, G2, R, Qr, Trans, P2, knn_idx, knn_dist, smpl_mode):
    """
    Args:
        X1: (M,3) means Pose1
        Q1: (M,4) orientation quats Pose1
        S1: (M,3) scales (stddev along axes)
        G1: (B,N,3) graph Pose1
        G2: (B,N,3) graph Pose2
        R:  (B,N,3,3) rotations
        Qr: (B,N,4) rotation quats
        theta: (B,N,3,3) rotation quats
        knn_idx: (B,M,K)
        knn_dist: (B,M,K)
    Returns:
        X2: (M,3) deformed means
        Q2: (M,4) deformed orientations
        S2: (M,3) scales (unchanged)
        Sigma2: (M,3,3) covariance matrices
    """
    sigma2 = knn_dist.mean()
    weights = torch.exp(-knn_dist / (2*sigma2)) #
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # (B,M,K)

    ## gather node positions
    G1_neigh   = pointutils.grouping_operation(G1.transpose(1,2).contiguous(), knn_idx)[0].permute(1,2,0)  # (M,K,3)
    G2_neigh   = pointutils.grouping_operation(G2.transpose(1,2).contiguous(), knn_idx)[0].permute(1,2,0)  # (M,K,3)
    
    ## gather node rotations
    B, N, _, _ = R.shape
    Rv = R.reshape(B,N,-1)
    R_neigh = pointutils.grouping_operation(Rv.transpose(1,2).contiguous(), knn_idx)[0].permute(1,2,0)
    R_neigh = R_neigh.reshape(R_neigh.shape[0], R_neigh.shape[1],3,3)  # (M,K,3,3)
    Qr_neigh = pointutils.grouping_operation(Qr.transpose(1,2).contiguous(), knn_idx)  # (B,4,N,K)
    Qr_neigh = Qr_neigh[0].permute(1,2,0)  # (M,K,4)

    ## means deformation
    if Trans is not None and smpl_mode:
        Trans_neigh = pointutils.grouping_operation(Trans.transpose(1,2).contiguous(), knn_idx)[0].permute(1,2,0)  # (M,K,3)
        X1r = torch.einsum('mkij,mkj->mki', R_neigh, X1.unsqueeze(1))
        contrib  = X1r + Trans_neigh
    else:
        d = X1.unsqueeze(1) - G1_neigh # (M,K,3)
        d_rot = torch.einsum('mkij,mkj->mki', R_neigh, d)  # (M,K,3)
        contrib = G2_neigh + d_rot# (M,K,3)

    X2 = (weights.unsqueeze(-1) * contrib).sum(dim=1)  # (M,3)
    
   
    ## quaternion deformation v1
    Q1_expand = Q1.unsqueeze(1).expand_as(Qr_neigh)
    dot = (Qr_neigh * Q1_expand).sum(-1, keepdim=True)
    Qr_neigh = torch.where(dot < 0, -Qr_neigh, Qr_neigh)  
    Qj = quaternion_multiply(Qr_neigh, Q1_expand)
    Q_weighted = (weights.unsqueeze(-1) * Qj).sum(dim=1)
    # if R_cloth is not None:
    #     theta_Qr = matrix_to_quaternion(R_cloth_neigh) 
    #     Qj = quaternion_multiply(theta_Qr, Qj)
    Q2 = F.normalize(Q_weighted, dim=-1, eps=1e-8)
    # Q2 = F.normalize(Q_weighted[:, 0, :], dim=-1)

    ## scale
    if S1 is not None:
        S2 = S1.clone()
    else:
        S2 = None

    return X2, Q2, S2

def skew(omega: torch.Tensor) -> torch.Tensor:
    """omega: (M,3,1) or (M,3) -> (M,3,3)"""
    if omega.ndim == 3:
        omega = omega.squeeze(-1)  # (M,3)
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    O = torch.zeros(omega.shape[0], 3, 3, device=omega.device, dtype=omega.dtype)
    O[:, 0, 1] = -wz
    O[:, 0, 2] =  wy
    O[:, 1, 0] =  wz
    O[:, 1, 2] = -wx
    O[:, 2, 0] = -wy
    O[:, 2, 1] =  wx
    return O

def so3_exp(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    omega: (M,3,1) or (M,3)
    return: (M,3,3) = exp([omega]_x)
    """
    if omega.ndim == 3:
        omega_vec = omega.squeeze(-1)  # (M,3)
    else:
        omega_vec = omega

    M = omega_vec.shape[0]
    device, dtype = omega_vec.device, omega_vec.dtype

    theta2 = (omega_vec * omega_vec).sum(dim=-1, keepdim=True)      # (M,1)
    theta = torch.sqrt(theta2 + eps)                                 # (M,1)

    W = skew(omega_vec)                                              # (M,3,3)
    W2 = W @ W                                                       # (M,3,3)

    # sin(theta)/theta and (1-cos(theta))/theta^2 with small-angle handling
    small = theta2 < 1e-6

    # default (non-small)
    A = torch.sin(theta) / theta                                     # (M,1)
    B = (1.0 - torch.cos(theta)) / (theta2 + eps)                    # (M,1)

    # taylor for small theta: 
    # A ≈ 1 - θ^2/6 + θ^4/120
    # B ≈ 1/2 - θ^2/24 + θ^4/720
    theta4 = theta2 * theta2
    A_t = 1.0 - theta2 / 6.0 + theta4 / 120.0
    B_t = 0.5 - theta2 / 24.0 + theta4 / 720.0

    A = torch.where(small, A_t, A)
    B = torch.where(small, B_t, B)

    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(M, 3, 3)
    R = I + A.view(M,1,1) * W + B.view(M,1,1) * W2
    return R


def compute_vertex_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10

    vertex_normals = np.zeros_like(vertices)

    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-10

    return vertex_normals, face_normals


def sample_points_on_mesh(vertices, faces, n_samples=8000, device="cuda"):
    """
    Uniform Sample (GPU)
    vertices: (N,3) torch.Tensor
    faces: (F,3) torch.LongTensor
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_areas = torch.linalg.norm(torch.cross(v1 - v0, v2 - v0), dim=1) / 2
    face_probs = face_areas / face_areas.sum()

    chosen_faces = torch.multinomial(face_probs, n_samples, replacement=True)
    f0, f1, f2 = v0[chosen_faces], v1[chosen_faces], v2[chosen_faces]

    u = torch.rand(n_samples, 1, device=device)
    v = torch.rand(n_samples, 1, device=device)
    mask = (u + v > 1)
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]

    samples = f0 + u * (f1 - f0) + v * (f2 - f0)
    return samples  # (n_samples, 3)


def balance_downsample_fixed_completion(observed_points, completion_points, target_total=8000, comp_fixed=1000, knn_dists=None, use_dist=False, device="cuda"):
    """
    observed_points: (N_obs, 3)
    completion_points: (N_comp, 3)
    target_total: 
    comp_fixed:
    """
    num_obs = observed_points.shape[0]
    num_comp = completion_points.shape[0]

    num_comp_keep = min(comp_fixed, num_comp)

    num_obs_keep = target_total - num_comp_keep
    num_obs_keep = min(num_obs_keep, num_obs)

    smpl_obs_mask = torch.zeros((num_obs_keep, 1), dtype=torch.bool, device=device)
    smpl_comp_mask = torch.ones((num_comp_keep, 1), dtype=torch.bool, device=device)

    if use_dist and knn_dists is not None:
        pass
    else:
        if num_obs_keep < num_obs:
            perm_obs = torch.randperm(num_obs, device=device)[:num_obs_keep]
            observed_points = observed_points[perm_obs]
            smpl_obs_mask = smpl_obs_mask[perm_obs]

    if num_comp_keep < num_comp:
        perm_comp = torch.randperm(num_comp, device=device)[:num_comp_keep]
        completion_points = completion_points[perm_comp]
        smpl_comp_mask = smpl_comp_mask[perm_comp]

    dense_points = torch.cat([observed_points, completion_points], dim=0)
    smpl_mask = torch.cat([smpl_obs_mask, smpl_comp_mask], dim=0)
    
    return dense_points, smpl_mask

def add_gaussian_noise_to_points(points: torch.Tensor, num_new_points: int, std_dev: float = 0.1) -> torch.Tensor:
    """
    Args:
        points (torch.Tensor):  (N, 3)
        num_new_points (int): 
        std_dev (float):

    Returns:
        torch.Tensor: (num_new_points, 3)
    """
    if points.numel() == 0:
        return None
    device = points.device
    
    random_indices = torch.randint(0, points.shape[0], (num_new_points,), device=device)
    base_points = points[random_indices]

    noise = torch.randn_like(base_points) * std_dev

    new_points = base_points + noise

    dists,idx = pointutils.three_nn(new_points.unsqueeze(0).contiguous(), points.unsqueeze(0).contiguous())
    dists[dists < 1e-6] = 1e-6
    weight = 1.0 / dists
    weight = weight / torch.sum(weight, -1,keepdim = True)   # [B,N,3]
    N = new_points.shape[0]
    # M = points.shape[0]
    
    nn_obj = 1.0 * pointutils.grouping_operation(points.unsqueeze(0).transpose(2,1).contiguous(), idx) #[B,C,N,3]
    interpolated_obj = torch.sum(nn_obj * weight.view(1, 1, N, 3), dim = -1) # [B,C,N,3]
    interpolated_obj = interpolated_obj.squeeze(0).transpose(1,0)
    
    return interpolated_obj#.detach().cpu().numpy()

def smpl_based_completion_and_densification(
    smpl_vertices, smpl_faces, raw_pointcloud, obj_flow=None,
    n_samples=None, k=3, threshold=0.05, hinge=False, device="cuda", fusion_mode=True
):
    """
    smpl_vertices: (6890,3) torch.Tensor
    smpl_faces: (13776,3) torch.LongTensor
    raw_pointcloud: (M,3) torch.Tensor
    n_samples: the number of dense points
    k: nn number
    threshold: the threshold of observed points
    """
    smpl_vertices = smpl_vertices.to(device)
    smpl_faces = smpl_faces.to(device)
    raw_pointcloud = raw_pointcloud.to(device)

    # 1. Densify SMPL Points
    if n_samples is not None and n_samples > 6890:
        smpl_samples = sample_points_on_mesh(smpl_vertices, smpl_faces, n_samples, device=device)
    else:
        smpl_samples = smpl_vertices

    if not fusion_mode:
        points_ids = torch.zeros(smpl_samples.shape[0], dtype=torch.int8, device=device)
        smpl_mask = torch.ones((smpl_samples.shape[0], 1), dtype=torch.bool, device=device)
        return smpl_samples, smpl_mask, points_ids  # (n_samples, 3)
    
    min_z = torch.min(raw_pointcloud[:,2])   # (N,)  
    valid_z = min_z + 0.1                    # (N,)
    z = raw_pointcloud[:, 2]                        # (N,)
    valid_mask = z >= valid_z                       # (N,)
    raw_pointcloud = raw_pointcloud[valid_mask]           # (M,3)

    mins, _ = smpl_vertices.min(dim=0)  # (3,)
    maxs, _ = smpl_vertices.max(dim=0)  # (3,)
    mask = (raw_pointcloud >= mins) & (raw_pointcloud <= maxs)  # (N,3) bool
    inside_mask = mask.all(dim=1)        
    nonrigid_pointcloud = raw_pointcloud[inside_mask]               # (N,)

    knn = knn_points(raw_pointcloud[None, ...], nonrigid_pointcloud[None, ...], K=1)  # 点云到控制节点
    dists, idx = knn.dists[0], knn.idx[0]  # (n_samples, k)
    avg_dist = torch.mean(dists, dim=1)
    mask_nonrigid = (avg_dist < threshold*1.0)
    nonrigid_pointcloud = raw_pointcloud[mask_nonrigid]   

    # visualize_point_cloud([raw_pointcloud[mask_nonrigid], raw_pointcloud[~mask_nonrigid]], [[1,0,0],[0,0,1], [0,1,0],[0.5, 0.5,0], [0.5, 0, 0.5]], show_coordinate_frame=True)
    
    # 2.2 Project X to SMPL 
    knn = knn_points(nonrigid_pointcloud[None, ...], smpl_samples[None, ...], K=3)
    dists, idx = knn.dists[0], knn.idx[0] 
    nn_ind_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)
    knn_verts = torch.gather(smpl_samples.unsqueeze(1).expand(-1, k, -1), 0, nn_ind_expanded) # (smpl_points_num, knn_neighbors, 3)

    weights = 1.0 / (dists + 1e-6)  # (B, N, k)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    new_raw_pts = (weights.unsqueeze(-1) * knn_verts).sum(dim=-2)  # (B, N, 3)

    # 3. Determine which areas are being observed
    knn = knn_points(smpl_samples[None, ...], new_raw_pts[None, ...], K=3)
    dists, idx = knn.dists[0], knn.idx[0]  # (n_samples, k)
    # avg_dist = torch.mean(dists, dim=1)
    # mask_observed = (avg_dist < threshold*0.1)
    mask_observed = (dists[:, 0] < threshold*0.1)
    
    # 4. Within the observation area, calculate the offset (average of k neighbors).
    nearest_points = nonrigid_pointcloud[idx]  # (n_samples, k, 3)
    offsets = nearest_points.mean(dim=1) - smpl_samples  # (n_samples, 3)

    # Extract feet points
    feet_points = smpl_vertices[feet_vids]   # [N_left, 3]
    observed_points = smpl_samples[mask_observed] + offsets[mask_observed]
    completion_points = smpl_samples[~mask_observed]

    if n_samples is not None and n_samples > 6890:
        target_total = n_samples
        comp_fixed = target_total - observed_points.shape[0]
    else:
        target_total = 6890
        comp_fixed = target_total - raw_pointcloud.shape[0]
    
    dense_points, smpl_mask = balance_downsample_fixed_completion(observed_points, completion_points, target_total=n_samples, comp_fixed=comp_fixed)

    knn = knn_points(dense_points[None, ...], feet_points[None, ...], K=1)
    dists, idx = knn.dists[0], knn.idx[0]  # (n_samples, k)
    # avg_dist = torch.mean(dists, dim=1)
    mask_feet = (dists[:, 0] > 0.0003)
    dense_points = torch.cat([dense_points[mask_feet], feet_points], dim=0) 
    feet_mask = torch.ones((feet_points.shape[0], 1), dtype=torch.bool, device=device)
    # feet_mask2 = torch.zeros((dense_points.shape[0], 1), dtype=torch.bool, device=device)
    # feet_mask2[-feet_mask.shape[0]:, 0] = True
    smpl_mask = torch.cat([smpl_mask[mask_feet], feet_mask], dim=0) 
    # visualize_point_cloud([dense_points, raw_pointcloud], [[1,0,0],[0,0,1], [0,1,0],[0.5, 0.5,0], [0.5, 0, 0.5]], show_coordinate_frame=True)
    
    # # Optional points
    knn = knn_points(smpl_samples[None, ...], dense_points[None, ...], K=1)
    dists, idx = knn.dists[0], knn.idx[0]  # (n_samples, k)
    res_mask = torch.logical_and(dists[:, 0] < 0.0007, dists[:, 0] > 0.0003)
    dense_points = torch.cat([dense_points, smpl_samples[res_mask]])
    smpl_mask = torch.cat([smpl_mask, res_mask[res_mask][:, None]], dim=0) 
    
    points_ids = torch.zeros(dense_points.shape[0], dtype=torch.int8, device=device)
    if hinge:
        hinge_pts = raw_pointcloud[~mask_nonrigid]
        if hinge_pts.shape[0] < 70:
            num_dense_pc = 150
        else:
            num_dense_pc = hinge_pts.shape[0] * 2
        dense_hinge_pts = add_gaussian_noise_to_points(hinge_pts, num_dense_pc, std_dev=0.1) 
        dense_points = torch.cat([dense_points, dense_hinge_pts])
        hinge_mask = torch.zeros((dense_hinge_pts.shape[0], 1), dtype=torch.bool, device=device)
        smpl_mask = torch.cat([smpl_mask, hinge_mask], dim=0) 
        hinge_points_ids = torch.ones(dense_hinge_pts.shape[0], dtype=torch.int8, device=device)
        points_ids = torch.cat([points_ids, hinge_points_ids], dim=0) 

    return dense_points, smpl_mask, points_ids  # (n_samples, 3)

def convert_tensor_to_pcd(points, file_name=None):
    points_np = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    if file_name is not None:
        output_file = file_name
        np.savetxt(output_file, points_np, fmt='%.3f') 
        print(f"The array has been successfully written to the file: {output_file}")
    return pcd

def invert_se3(G):
    """
    G: (...,4,4) rigid transform with R orthonormal
    returns G^{-1}
    """
    R = G[..., :3, :3]
    t = G[..., :3, 3:4]
    Rt = R.transpose(-1, -2)
    tinv = -Rt @ t
    Ginv = torch.zeros_like(G)
    Ginv[..., :3, :3] = Rt
    Ginv[..., :3, 3:4] = tinv
    Ginv[..., 3, 3] = 1.0
    return Ginv

def project_to_so3(M):
    """
    M: (...,3,3) general matrices
    returns R in SO(3) via SVD projection
    """
    U, S, Vh = torch.linalg.svd(M)  # Vh is V^T
    R = U @ Vh
    # Fix improper rotation (det=-1)
    det = torch.det(R)
    # If det < 0, flip last column of U (or last row of Vh)
    mask = det < 0
    if mask.any():
        U2 = U.clone()
        U2[mask, :, -1] *= -1.0
        R = U2 @ Vh
    return R

def per_vertex_rt_from_relative_joints(G1, G2, W):
    """
    G1, G2: (K,4,4) joint global transforms for pose1 and pose2
    W: (V,K) skinning weights
    X1, X2: (V,3) vertices/points in pose1 and pose2 (optional)
    Returns:
      R: (V,3,3)
      t: (V,3)
      T_affine: (V,4,4) (mixed affine before projection)
    """
    # device = G1.device
    # K = G1.shape[0]
    # V = W.shape[0]

    # 1) relative joint transforms ΔG_k = G2_k * inv(G1_k)
    dG = G2 @ invert_se3(G1)  # (K,4,4)

    # 2) per-vertex mixed transform T_v = sum_k w_vk * dG_k
    # Efficient einsum: (V,K) x (K,4,4) -> (V,4,4)
    # T = torch.einsum('vk,kij->vij', W, dG)
    T = torch.einsum("bnj, bjrc -> bnrc", W, dG)

    # 3) project linear part to SO(3)
    M = T[:, :, :3, :3]                  # (V,3,3)
    R = project_to_so3(M)             # (V,3,3)

    # 4) translation
    # if use_point_consistent_t:
    #     if X1 is None or X2 is None:
    #         raise ValueError("X1 and X2 required when use_point_consistent_t=True")
    #     t = X2 - torch.einsum('vij,vj->vi', R, X1)  # (V,3)
    # else:
    t = T[:,:, :3, 3]  # (V,3)

    return R, t, T


def normalizeVec(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def compute_vertex_normals(P, faces):
    """
    P: (N, V, 3) P1
    faces: (F, 3) long, SMPL topology
    return: (N, V, 3) vertex, normal, n₁,i
    """
    N, V, _ = P.shape
    F_ = faces.shape[0]
    device = P.device
    faces = faces.to(device)

    v0 = P[:, faces[:, 0], :]  # (N, F, 3)
    v1 = P[:, faces[:, 1], :]
    v2 = P[:, faces[:, 2], :]

    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (N, F, 3)
    fn = normalizeVec(fn)

    vn = torch.zeros_like(P)  # (N, V, 3)
    vn.index_add_(1, faces[:, 0], fn)
    vn.index_add_(1, faces[:, 1], fn)
    vn.index_add_(1, faces[:, 2], fn)
    vn = normalizeVec(vn)
    return vn


def axis_normal_angle_to_matrix(axis, angle):
    """
    axis: (..., 3) 
    angle: (...)   rad
    return: (..., 3, 3) rotation matrix
    """
    axis = axis[..., None]                           # (..., 3, 1)
    x, y, z = axis[..., 0, 0], axis[..., 1, 0], axis[..., 2, 0]

    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c

    r00 = c + x * x * one_c
    r01 = x * y * one_c - z * s
    r02 = x * z * one_c + y * s

    r10 = y * x * one_c + z * s
    r11 = c + y * y * one_c
    r12 = y * z * one_c - x * s

    r20 = z * x * one_c - y * s
    r21 = z * y * one_c + x * s
    r22 = c + z * z * one_c

    R = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )
    return R


def compute_body_to_cloth_angle(
    P1,
    X1,
    faces,
    k_cloth=8,
    ):
    """
    Calculate the rotation angle θᵢ from P1 to X1 around the normal n₁,i of vertex P1, and the corresponding rotation matrix ΔR_cloth_i(θᵢ).

    Input:
        P1: (N, Vp, 3)  SMPL
        X1: (N, Vc, 3)  
        faces: (F, 3)   
        k_cloth: int   

    Return:
        theta: (N, Vp)         Rotation angle (rad) about the normal n₁,i
        n_body: (N, Vp, 3)     Vertex Normal n₁,i
        R: (N, Vp, 3, 3)       Rotation matrix around n₁,i ΔR_cloth_i(θᵢ)
    """
    device = P1.device
    N, Vp, _ = P1.shape
    _, Vc, _ = X1.shape

    # 1) compute n₁,i
    n_body = compute_vertex_normals(P1, faces)  # (N, Vp, 3)

    # 2) knn
    _, knn_idx, _ = knn_points(P1, X1, K=k_cloth)
    knn_pts = knn_gather(X1, knn_idx)
    x_target = knn_pts.mean(dim=2)  # (N, Vp, 3)

    # 3) relative vector
    w = x_target - P1  # (N, Vp, 3)

    # 4) orthogonal basis vector (t1_i, t2_i, n_i)
    n = normalizeVec(n_body)  # (N, Vp, 3)


    global_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    up = global_up.view(1, 1, 3).expand(N, Vp, 3)
    #check
    cos_n_up = (n * up).sum(-1, keepdim=True).abs()  # (N, Vp, 1)
    alt_axis = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 1, 3)
    up = torch.where(
        cos_n_up > 0.99,
        alt_axis.expand_as(up),
        up,
    )

    # t1 = up × n，t2 = n × t1
    t1 = normalizeVec(torch.cross(up, n, dim=-1))  # (N, Vp, 3)
    t2 = normalizeVec(torch.cross(n, t1, dim=-1))  # (N, Vp, 3)

    # 5) Project w onto the tangent plane: w_t = w - <w,n> n
    w_n = (w * n).sum(-1, keepdim=True) * n
    w_t = w - w_n  # (N, Vp, 3)

    # 6) Get the polar angle in the (t1, t2) basis.：w_t = a * t1 + b * t2
    a = (w_t * t1).sum(-1)  # (N, Vp)
    b = (w_t * t2).sum(-1)  # (N, Vp)

    # θ = atan2(b, a)
    theta = torch.atan2(b, a)  # (N, Vp), Angle around normal n

    # 7) Get the rotation matrix around n, ΔR_cloth_i(θᵢ)
    R = axis_normal_angle_to_matrix(n, theta)  # (N, Vp, 3, 3)

    theta = n * theta[..., None]

    return theta, n_body, R



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "/home/ubuntu/Repositories/drivestudio/01/"
    
    # scene_rot_files = os.listdir(os.path.join(data_path,'flow/rot/'))
    scene_rot_files = glob.glob(os.path.join(data_path, '*.npz'))
    scene_rot_files.sort()

    pattern = r"/(\d+)_(\d+)\.npz$"

    device = "cuda:0"

    for ins_i, item in enumerate(scene_rot_files):
        if ins_i < 4:
            continue
        # 使用 re.search 进行匹配
        match = re.search(pattern, item)
        ref_fi = int(match.group(1))
        ins_id = int(match.group(2))
        cur_scene = np.load(scene_rot_files[ins_i], allow_pickle=True)
        cur_ins_ids = cur_scene["ins_id"]
        cur_origin_full_pc = cur_scene["sub_src"]
        cur_ins_quats = torch.from_numpy(cur_scene["quats"]).cuda()
        cur_ins_trans = torch.from_numpy(cur_scene["trans"]).cuda()
        init_beta = torch.from_numpy(cur_scene["betas"])
        
        deformed_can_means, R, deformed_can_means_faces, skeleton_can = get_smpl_template(num_human=1, init_beta=init_beta.unsqueeze(0), instances_quats=cur_ins_quats[ref_fi], instances_trans=cur_ins_trans[ref_fi], smpl_points_num=6890, smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl", use_canical=False)
        deformed_can_means = deformed_can_means.cuda()
        skeleton_can = skeleton_can.cuda()
        deformed_can_means_faces = deformed_can_means_faces.cuda()

        # o2w = torch.from_numpy(cur_scene["o2w"]).cuda() # (num_instances, 4)
        o2w_dict = cur_scene["o2w"]
        o2w_dict = {k: torch.from_numpy(v).to(device=device, dtype=torch.float32) for k, v in o2w_dict[0].items()}
        first_key = next(iter(o2w_dict))
        ref_o2w = o2w_dict[ref_fi]
        ref_w2o = torch.inverse(ref_o2w)
        # trans_per_pts = cur_ins_trans
        # smpl_verts_w2o = (deformed_can_means - trans_per_pts.unsqueeze(0)) @ rot_cur_frame_inv.T
        # skeleton_can_w2o = (skeleton_can - trans_per_pts.unsqueeze(0)) @ rot_cur_frame_inv.T
        smpl_verts_w2o = transform_points(deformed_can_means[0], ref_w2o)
        smpl_verts_w2o = smpl_verts_w2o.unsqueeze(0).contiguous()
        skeleton_can_w2o = transform_points(skeleton_can[0], ref_w2o)
        skeleton_can_w2o = skeleton_can_w2o.unsqueeze(0).contiguous()
        cur_sub_src = torch.from_numpy(cur_origin_full_pc).cuda()

        # for fi in range(first_key, cur_ins_quats.shape[0]):

        fi = ref_fi - 1
        if fi == -1:
            fi = 1
        if fi >= cur_ins_quats.shape[0]:
            fi = cur_ins_quats.shape[0] - 2
        # smpl_infos = np.load(scene_rot_files[fi], allow_pickle=True)
        cur_ins_quats2 = cur_ins_quats[fi]#torch.from_numpy(smpl_infos["quats"])
        cur_ins_trans2 = cur_ins_trans[fi]#torch.from_numpy(smpl_infos["trans"])
        # init_beta = torch.from_numpy(smpl_infos["betas"])

        ######################################
        deformed_can_means2, _, deformed_can_means_faces2, skeleton_can2 = get_smpl_template(num_human=1, init_beta=init_beta.unsqueeze(0), instances_quats=cur_ins_quats2, instances_trans=cur_ins_trans2, smpl_points_num=6890, smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl", use_canical=False)
        deformed_can_means2 = deformed_can_means2.cuda()
        skeleton_can2 = skeleton_can2.cuda()
        deformed_can_means_faces2 = deformed_can_means_faces2.cuda()

        o2w = o2w_dict[fi]
        w2o = torch.inverse(o2w)
        smpl_verts_w2o2 = transform_points(deformed_can_means2[0], w2o)
        smpl_verts_w2o2 = smpl_verts_w2o2.unsqueeze(0).contiguous()
        skeleton_can_w2o2 = transform_points(skeleton_can2[0], w2o)
        skeleton_can_w2o2 = skeleton_can_w2o2.unsqueeze(0).contiguous()
        #####################################
        # cur_sub_src = torch.from_numpy(cur_origin_full_pc[i]).cuda()
        # visualize_point_cloud([cur_sub_src], [[1,0,0],[0,0,1], [0,1,0],[0.5, 0.5,0], [0.5, 0, 0.5]], show_coordinate_frame=False)

        if ins_i == 6:
            _, source_cuda = bcpdreg(smpl_verts_w2o[0].detach().cpu().numpy(), cur_sub_src[0].detach().cpu().numpy(), normalization_type='n', quiet_mode=True, show_fov=False)
        else:
            _, source_cuda = bcpdreg(smpl_verts_w2o[0].detach().cpu().numpy(), cur_sub_src[0].detach().cpu().numpy(), normalization_type='e', quiet_mode=True, show_fov=False)
        # new_smpl.paint_uniform_color([1,0,0])
        # target.paint_uniform_color([0,0,1])
        # source.paint_uniform_color([0,1,0])
        source_cuda = torch.from_numpy(source_cuda).to(dtype=torch.float32).cuda()
        # visualize_point_cloud([source_cuda, cur_sub_src[0]], [[1,0,0],[0,0,1], [0,1,0],[0.5, 0.5,0], [0.5, 0, 0.5]], show_coordinate_frame=True)
        visualize_point_cloud([source_cuda, smpl_verts_w2o2[0], cur_sub_src[0]], [[1,0,0],[0,0,1], [0,1,0],[0.5, 0.5,0], [0.5, 0, 0.5]], show_coordinate_frame=True)
        ######################

        # deformation_graph_pipeline(smpl_verts_w2o[0],smpl_verts_w2o2[0], cur_sub_src)
        dense_pc = smpl_based_completion_and_densification(source_cuda, deformed_can_means_faces, cur_sub_src[0], n_samples=15000, k=5, threshold=0.002, device=device)
        # sampled_idx = (torch.randperm(dense_pc.shape[0])[:8000])
        # dense_pc = dense_pc[sampled_idx]
        X_quats=torch.zeros([dense_pc.shape[0], 4], dtype=torch.float32).cuda()
        X_quats[:,0] = 1.0
        X2 = deformation_graph_pipeline(source_cuda, smpl_verts_w2o2[0], dense_pc, X_quats=X_quats, n_nodes=1500, K_point=5)
        visualize_point_cloud([cur_sub_src[0], X2], [[1,0,0],[0,0,1], [0,1,0],[0.5, 0.5,0], [0.5, 0, 0.5]], show_coordinate_frame=True)
            
        


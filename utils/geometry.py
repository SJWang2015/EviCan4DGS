# Utility functions for geometric transformations and projections.
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

def interpolate_matrix_and_quaternion(fraction, matrix1=None, matrix2=None, q1=None, q2=None, eps=1e-8):
    """
    使用四元数 Slerp 在两组旋转矩阵之间插值，同时返回插值矩阵和四元数。
    
    参数:
        matrix1: (B, 3, 3)
        matrix2: (B, 3, 3)
        q1: (B, 4)
        q2: (B, 4)
        fraction: 
            - 标量 (float or 0-dim tensor)，对所有 batch 使用同一插值系数
            - 或 (B,) / (B,1)，每个 batch 各自一个插值系数
    返回:
        R_interp: (B, 3, 3) 插值后的旋转矩阵
        q_interp: (B, 4)    插值后的单位四元数 [w, x, y, z]
    """
    # 1. 矩阵 -> 四元数
    if q1 is None or q2 is None:
        q1 = matrix_to_quaternion(matrix1)  # (B, 4)
        q2 = matrix_to_quaternion(matrix2)  # (B, 4)


    # 归一化，确保是单位四元数
    q1 = F.normalize(q1, dim=-1, eps=eps)
    q2 = F.normalize(q2, dim=-1, eps=eps)

    B = q1.size(0)
    device = q1.device
    dtype = q1.dtype

    # 2. 处理 fraction 形状
    if not torch.is_tensor(fraction):
        fraction = torch.tensor(fraction, device=device, dtype=dtype)
    if fraction.dim() == 0:
        fraction = fraction.expand(B)          # (B,)
    if fraction.dim() == 1:
        fraction = fraction.unsqueeze(-1)      # (B,1)
    # 此时 fraction: (B,1)

    # 3. 点积和最短弧处理
    dot = (q1 * q2).sum(dim=-1, keepdim=True)   # (B,1)
    dot = torch.clamp(dot, -1.0, 1.0)

    neg_mask = dot < 0
    q2 = torch.where(neg_mask, -q2, q2)         # 反号以走最短弧
    dot = torch.where(neg_mask, -dot, dot)

    # 4. 小角度与一般情况分支
    similar_mask = dot > 0.9995  # (B,1) 布尔

    # 4.1 小角度 -> 直接 Lerp + 归一化
    q_interp_similar = q1 + fraction * (q2 - q1)     # (B,4)
    q_interp_similar = F.normalize(q_interp_similar, dim=-1, eps=eps)

    # 4.2 一般情况 -> 标准 Slerp
    theta_0 = torch.acos(dot)                        # (B,1)
    theta = theta_0 * fraction                       # (B,1)

    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)

    # 避免除零
    sin_theta_0 = torch.where(
        sin_theta_0.abs() < eps,
        torch.full_like(sin_theta_0, eps),
        sin_theta_0
    )

    # 你的公式：s1 = cosθ - cosθ0 * sinθ / sinθ0
    #          s2 = sinθ / sinθ0
    s1 = torch.cos(theta) - dot * sin_theta / sin_theta_0  # (B,1)
    s2 = sin_theta / sin_theta_0                           # (B,1)

    q_interp_general = s1 * q1 + s2 * q2                   # (B,4)
    q_interp_general = F.normalize(q_interp_general, dim=-1, eps=eps)

    # 5. 根据 similar_mask 选择对应结果
    similar_mask_flat = similar_mask.squeeze(-1)           # (B,)
    q_interp = torch.empty_like(q1)
    q_interp[similar_mask_flat] = q_interp_similar[similar_mask_flat]
    q_interp[~similar_mask_flat] = q_interp_general[~similar_mask_flat]

    # 6. 再归一化一次，确保是单位四元数
    q_interp = F.normalize(q_interp, dim=-1, eps=eps)

    # 7. 四元数 -> 旋转矩阵
    R_interp = quaternion_to_matrix(q_interp)              # (B,3,3)

    return R_interp, q_interp

@torch.jit.script
def slerp_coefficients(dot: torch.Tensor, fraction: torch.Tensor, eps: float = 1e-8):
    """JIT 编译的 slerp 系数计算"""
    theta_0 = torch.acos(dot)
    theta = theta_0 * fraction
    sin_theta_0 = torch.sin(theta_0)
    
    use_lerp = sin_theta_0.abs() < eps
    sin_theta_0_safe = torch.where(use_lerp, torch.ones_like(sin_theta_0), sin_theta_0)
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    s1_slerp = cos_theta - dot * sin_theta / sin_theta_0_safe
    s2_slerp = sin_theta / sin_theta_0_safe
    
    s1 = torch.where(use_lerp, 1.0 - fraction, s1_slerp)
    s2 = torch.where(use_lerp, fraction, s2_slerp)
    
    return s1, s2


def interpolate_matrix_and_quaternion_jit(fraction, matrix1=None, matrix2=None, q1=None, q2=None, eps=1e-8):
    """使用 JIT 编译核心计算的版本"""
    if q1 is None or q2 is None:
        q1 = matrix_to_quaternion(matrix1)
        q2 = matrix_to_quaternion(matrix2)

    q1 = F.normalize(q1, dim=-1, eps=eps)
    q2 = F.normalize(q2, dim=-1, eps=eps)

    B = q1.size(0)
    device, dtype = q1.device, q1.dtype

    if not torch.is_tensor(fraction):
        fraction = torch.tensor(fraction, device=device, dtype=dtype)
    fraction = fraction.view(-1, 1).expand(B, 1)

    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    sign = dot.sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q2 = q2 * sign
    dot = (dot * sign).clamp(0.0, 1.0)

    s1, s2 = slerp_coefficients(dot, fraction, eps)
    
    q_interp = F.normalize(s1 * q1 + s2 * q2, dim=-1, eps=eps)
    R_interp = quaternion_to_matrix(q_interp)

    return R_interp, q_interp

class QuaternionLossDot(nn.Module):
    """
    基于 1 - |<q_pred, q_gt>| 的四元数 loss。
    - 四元数格式: wxyz
    - 自动归一化 q_pred 和 q_gt
    - 自动处理 q 和 -q 的等价性
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_quat, gt_quat):
        """
        pred_quat: [B, 4]，网络预测四元数（wxyz）
        gt_quat:   [B, 4]，标注四元数（wxyz）
        return:    标量 loss
        """
        # 归一化，保证是单位四元数
        pred_quat = pred_quat / (pred_quat.norm(dim=-1, keepdim=True) + self.eps)
        gt_quat   = gt_quat   / (gt_quat.norm(dim=-1, keepdim=True)   + self.eps)

        # 点积 <q_pred, q_gt>，形状 [B]
        dot = torch.sum(pred_quat * gt_quat, dim=-1)

        # q 和 -q 表示同一旋转，所以取绝对值
        loss = 1.0 - dot.abs()   # [B]

        # 返回 batch 平均
        return loss.mean()

# def transform_points(points, transform_matrix):
#     """
#     Apply a 4x4 transformation matrix to 3D points.

#     Args:
#         points: (N, 3) tensor of 3D points
#         transform_matrix: (4, 4) transformation matrix

#     Returns:
#         (N, 3) tensor of transformed 3D points
#     """
#     ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
#     homo_points = torch.cat([points, ones], dim=1)  # N x 4
#     transformed_points = torch.matmul(homo_points, transform_matrix.T)
#     return transformed_points[:, :3]
def transform_points(points, transform_matrix):
    """
    Apply 4x4 transformation matrix/matrices to 3D points.

    Args:
        points: (N, 3) or (B, N, 3) tensor of 3D points
        transform_matrix: (4, 4) or (B, 4, 4) transformation matrix/matrices

    Returns:
        (N, 3) or (B, N, 3) transformed 3D points
    """
    single_batch = False

    if points.dim() == 2:
        points = points.unsqueeze(0)   # [1,N,3]
        transform_matrix = transform_matrix.unsqueeze(0) \
            if transform_matrix.dim() == 2 else transform_matrix
        single_batch = True

    B, N, _ = points.shape

    ones = torch.ones((B, N, 1), dtype=points.dtype, device=points.device)
    homo_points = torch.cat([points, ones], dim=-1)
    
    transformed_points = torch.matmul(homo_points, transform_matrix.transpose(1,2))  # [B,N,4]

    result = transformed_points[:, :, :3]

    if single_batch:
        result = result.squeeze(0)

    return result

def get_corners(l: float, w: float, h: float):
    """
    Get 8 corners of a 3D bounding box centered at origin.

    Args:
        l, w, h: length, width, height of the box

    Returns:
        (3, 8) array of corner coordinates
    """
    return np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2],
    ])
    
def project_camera_points_to_image(points_cam, cam_intrinsics):
    """
    Project 3D points from camera space to 2D image space.

    Args:
        points_cam (np.ndarray): Shape (N, 3), points in camera space.
        cam_intrinsics (np.ndarray): Shape (3, 3), intrinsic matrix of the camera.

    Returns:
        tuple: (projected_points, depths)
            - projected_points (np.ndarray): Shape (N, 2), projected 2D points in image space.
            - depths (np.ndarray): Shape (N,), depth values of the projected points.
    """
    points_img = cam_intrinsics @ points_cam.T
    depths = points_img[2, :]
    projected_points = (points_img[:2, :] / (depths + 1e-6)).T
    
    return projected_points, depths

def cube_root(x):
    return torch.sign(x) * torch.abs(x) ** (1. / 3)

def spherical_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)

def uniform_sample_sphere(num_samples, device, inverse=False):
    """
    refer to https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    sample points uniformly inside a sphere
    """
    if not inverse:
        dist = torch.rand((num_samples,)).to(device)
        dist = cube_root(dist)
    else:
        dist = torch.rand((num_samples,)).to(device)
        dist = 1 / dist.clamp_min(0.02)
    thetas = torch.arccos(2 * torch.rand((num_samples,)) - 1).to(device)
    phis = 2 * torch.pi * torch.rand((num_samples,)).to(device)
    pts = spherical_to_cartesian(dist, thetas, phis)
    return pts

def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
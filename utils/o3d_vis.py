import os
import path
import open3d as o3d
import numpy as np
# import torch
from pybcpd import register as bcpd_register
from pybcpd import TransformationType

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.colors as mcolors

from plyfile import PlyData, PlyElement

def save_gaussians_as_ply(path, means, quats, scales, opacities, rgbs):

    N = means.shape[0]

    elements = np.empty(N, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])

    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1]
    elements['z'] = means[:, 2]

    elements['nx'] = 0
    elements['ny'] = 0
    elements['nz'] = 0

    elements['f_dc_0'] = rgbs[:, 0]
    elements['f_dc_1'] = rgbs[:, 1]
    elements['f_dc_2'] = rgbs[:, 2]

    elements['opacity'] = opacities[:, 0]

    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]

    elements['rot_0'] = quats[:, 0]
    elements['rot_1'] = quats[:, 1]
    elements['rot_2'] = quats[:, 2]
    elements['rot_3'] = quats[:, 3]

    ply = PlyData([PlyElement.describe(elements, 'vertex')])
    ply.write(path)

    print("Saved to", path)


def flow_to_color(flow):
    """光流转HSV伪彩色: 方向->色相, 大小->亮度"""
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()

    u, v = flow[...,0], flow[...,1]
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[...,0] = (ang + np.pi) / (2*np.pi)  # [0,1]
    hsv[...,1] = 1.0
    hsv[...,2] = mag / (mag.max() + 1e-6)
    return mcolors.hsv_to_rgb(hsv)

def visualize_two_flows(flow1, flow2, step=5):
    """
    对比两个光流场:
      - quiver 使用原始数值, 自动选择 scale
      - color 显示方向+强度
      - 差值热力图
    """
    if isinstance(flow1, torch.Tensor):
        flow1 = flow1.detach().cpu().numpy()
    if isinstance(flow2, torch.Tensor):
        flow2 = flow2.detach().cpu().numpy()

    H, W, _ = flow1.shape
    y, x = np.mgrid[0:H:step, 0:W:step]

    fig, axes = plt.subplots(3,2, figsize=(12,14))

    for idx, (flow, color, title_prefix) in enumerate(
        [(flow1, "r", "Flow 1"), (flow2, "b", "Flow 2")]
    ):
        u, v = flow[...,0], flow[...,1]
        mag = np.sqrt(u**2 + v**2)
        mean_mag = mag.mean()
        
        # 自动 scale: 让平均箭头长度 ≈ 1
        scale = max(mean_mag, 1e-6)

        # quiver (保留真实大小，只是缩放显示)
        axes[idx,0].quiver(
            x, y, u[::step,::step], -v[::step,::step],
            color=color, angles="xy", scale_units="xy", scale=scale
        )
        axes[idx,0].invert_yaxis()
        axes[idx,0].set_title(f"{title_prefix} (quiver, scaled)")

        # color
        axes[idx,1].imshow(flow_to_color(flow))
        axes[idx,1].set_title(f"{title_prefix} (color)")
        axes[idx,1].axis("off")

    # # 差异热力图 (使用真实差值)
    # diff = np.linalg.norm(flow1 - flow2, axis=-1)
    # im = axes[2,0].imshow(diff, cmap="magma")
    # axes[2,0].set_title("Flow Difference (‖flow1 - flow2‖)")
    # axes[2,0].axis("off")
    # plt.colorbar(im, ax=axes[2,0], fraction=0.046)

    # # 空位，可放别的图
    # axes[2,1].axis("on")

    plt.tight_layout()
    plt.show()

def bcpdreg(source, target, normalization_type='n', transformation_type=4, quiet_mode=True, show_fov=True):
    if transformation_type == 0:
        _transformation_type=TransformationType.TSRN # 全部变换（平移、缩放、旋转、非刚性）
    elif transformation_type == 1:
        _transformation_type=TransformationType.TAN # 仿射+非刚性
    elif transformation_type == 2:
        _transformation_type=TransformationType.TA # 仅仿射
    elif transformation_type == 3:
        _transformation_type=TransformationType.TSR # 相似性（平移、缩放、旋转）
    elif transformation_type == 4: 
        _transformation_type=TransformationType.TR # 刚性（平移、旋转）
    else:
        _transformation_type=TransformationType.TN  # 仅非刚性

    nystrom_J=350
    nystrom_K = 70

    if source.shape[0]+target.shape[0] <=50 or target.shape[0]<20 or source.shape[0] < 20:
        return None, None, None
    else:
        if source.shape[0]+target.shape[0] <= nystrom_J:
            nystrom_J = np.max(np.array([source.shape[0], target.shape[0]]))
            print("The M+N is:", source.shape[0] + target.shape[0], "; Reset nystrom_J to", nystrom_J)
        if target.shape[0] <= nystrom_K and target.shape[0] > 2:
            if target.shape[0] > int(nystrom_J/4):
                nystrom_K = int(nystrom_J/4)
            else:
                nystrom_K = target.shape[0]
            print("The M is:", target.shape[0], "; Reset nystrom_K to", nystrom_K)
        nystrom_K = np.clip(nystrom_K, 10, nystrom_J//4)
        if nystrom_K < 0 or nystrom_J < 0:
            return None, None, None
    
        # if nystrom_K < 70 or nystrom_J < 350:
        result = bcpd_register(target, source, random_seed=1,
                omega=0.0, lambda_param=25.0, gamma=15.0, beta=6.0,
                max_iter=300, min_iter=80, tolerance=1e-6, nystrom_J=nystrom_J, nystrom_K=nystrom_K, quiet_mode=quiet_mode, accel_mode=False, normalization_type=normalization_type, transformation_type=_transformation_type)

        
        pred_tgt = result['transformed_points']
        obj_flow = pred_tgt - source
    # show_fov = False
    if show_fov:
        vis_list = []
        pcd_src = o3d.geometry.PointCloud()
        # flow = result["displacement_field"]
        # Y_trans = Y + flow
        pcd_src.points = o3d.utility.Vector3dVector(source)
        pcd_src.paint_uniform_color([0,1.0,0])
        vis_list += [pcd_src]

        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(target)
        pcd_tgt.paint_uniform_color([1.0, 0,0])
        vis_list += [pcd_tgt]

        pcd_pred = o3d.geometry.PointCloud()
        # pc_pred = source + result['displacement_field']
        pcd_pred.points = o3d.utility.Vector3dVector(pred_tgt)
        pcd_pred.paint_uniform_color([0,0,1.0])
        vis_list += [pcd_pred]
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        vis_list.append(mesh_frame)
        o3d.visualization.draw_geometries(vis_list)

    return obj_flow, result['transformed_points'], result

def visualize_point_cloud(points_list: list, colors_list: list = None, faces=None, radius: float = 0.05, show_face=False, show_coordinate_frame=False, show_voxel=False) -> None:
    """
    Visualize a point cloud using Open3D.

    Args:
        points (torch.Tensor): A list of tensors with shape (N, 3) representing the point coordinates.
        colors (torch.Tensor, optional): A list of tensors with shape (N, 3) representing the RGB colors of the points.
        radius (float, optional): The radius of the points in the visualization. Default is 0.01.
    """
    if not isinstance(points_list, list):
        raise TypeError("points_list must be a list of tensors")
    
    if colors_list is not None and not isinstance(colors_list, list):
        raise TypeError("colors must be a list of tensors/numpy.array or None")
    
    
    vis_list = []
    for i, item in enumerate(points_list):
        points_np = item.cpu().detach().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        if colors_list is not None:
            if isinstance(colors_list[i], list):
                pcd.paint_uniform_color(colors_list[i])
            else:
                colors_np = colors_list[i].cpu().detach().numpy()
                pcd.colors = o3d.utility.Vector3dVector(colors_np)
        else:
            colors_np = np.ones_like(points_np)  # Default to white if no colors are provided
        
        vis_list.append(pcd)


        # fit to unit cube
        # N = points_np.shape[0]
        
        # pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))

        if show_voxel:
            pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
                center=pcd.get_center())
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                        voxel_size=radius)
            vis_list.append(voxel_grid)


        # else:
        #     vis_list.append(pcd)
        # o3d.visualization.draw_geometries([voxel_grid])
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(points_list[0].detach().cpu().numpy())
    # mesh.triangles = o3d.utility.Vector3iVector(faces)
    # vis_list.append(mesh)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, -3.4])
    if show_coordinate_frame:
        vis_list.append(mesh_frame)
    
    o3d.visualization.draw_geometries(vis_list, point_show_normal=False, mesh_show_back_face=True)
    # pcd.colors = o3d.utility.Vector3dVector(colors_np)
    # if colors_np.shape[0] == 1:
    #     pcd.paint_uniform_color(colors_np[0])
    # elif colors_np.shape[0] == 3:
    #     pcd.colors = o3d.utility.Vector3dVector(colors_np)
    # else:
    #     raise ValueError("colors_np2 must have shape (N, 3) or (1, 3)")

    # points_np2 = points_list[1].cpu().numpy()
    # if colors_list is not None:
    #     colors_np2 = colors_list[1]
    # else:
    #     colors_np2 = np.ones_like(points_np2)  # Default to white if no colors are provided

    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(points_np2) # src_pcs + src_flow
    # pcd2.paint_uniform_color(colors_np2)
    # # if colors_np2.shape[0] == 1:
    # #     pcd2.paint_uniform_color(colors_np2[0])
    # # elif colors_np2.shape[0] == 3:
    # #     pcd2.colors = o3d.utility.Vector3dVector(colors_np2)
    # # else:
    # #     raise ValueError("colors_np2 must have shape (N, 3) or (1, 3)")

    # points_np3 = points_list[2].cpu().numpy()
    # if colors_list is not None:
    #     colors_np3 = colors_list[2]
    # else:
    #     colors_np3 = np.ones_like(points_np3)  # Default to white if no colors are provided

    # pcd3 = o3d.geometry.PointCloud()
    # pcd3.points = o3d.utility.Vector3dVector(points_np3) # src_pcs + src_flow
    # pcd3.paint_uniform_color(colors_np3)


    

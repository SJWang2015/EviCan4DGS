#!/usr/bin/env python3
"""
使用示例：展示如何使用表面重建工具
"""

import torch
import numpy as np
# from pyTriangleMesh import triMesh
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from utils.pyTriangleMesh.surface_reconstruction import PedestrianSurfaceReconstructor
import warnings
from typing import Tuple, Dict, Optional, Union
# 示例用法 (假设你的点云数据)
# 假设你有 N 个点，每个点有 (x, y, z) 坐标
num_points = 1000
points_cpu = torch.randn(num_points, 3, dtype=torch.float32)

# 确保点数据在 CUDA 设备上
points_gpu = points_cpu.cuda()

algorithm_type = 0  # 0: 基于法向量的三角化, 1: Alpha Shape, 2: Ball Pivoting
param = 0.1         # 对应的算法参数 (例如 alpha 值或 ball_radius)



# 创建表面重建器
surface_reconstructor = PedestrianSurfaceReconstructor()

# 生成更密集的行人点云
def generate_dense_pedestrian_pointcloud(num_points=2000):
    """生成密集的行人点云"""
    points = []
    file_path = '/home/ubuntu/Repositories/drivestudio/output_root/cpddeform/PLYs/aggregated_instance_lidar_pts/ID=77_1.ply'
    o3d_points = o3d.io.read_point_cloud(file_path)
    points = np.asarray(o3d_points.points)
    file_path = '/home/ubuntu/Repositories/drivestudio/Y1.npy'
    points = np.load(file_path)
    # return torch.from_numpy(points).to(torch.float32)
    file_path = '/home/ubuntu/Repositories/drivestudio/xyz.npz'
    points = np.load(file_path)
    x = points['x']
    y = points['y']
    z = points['z']
    # color = points['color']
    # colors_np = np.hstack([color[:,i].flatten() for i in range(3)])
    np_points = np.vstack([x.flatten(), y.flatten(), z.flatten()])
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.paint_uniform_color([1,0,0])
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1)
    # o3d.visualization.draw_geometries([mesh], point_show_normal=False, mesh_show_back_face=True)
    # return torch.from_numpy(points).to(torch.float32)
    return torch.from_numpy(x).to(torch.float32), torch.from_numpy(y).to(torch.float32),torch.from_numpy(z).to(torch.float32), np_points
    # 头部（球体）
    # for _ in range(num_points//6):
    #     theta = np.random.uniform(0, 2*np.pi)
    #     phi = np.random.uniform(0, np.pi)
    #     r = 0.08 + 0.02 * np.random.random()
    #     x = r * np.sin(phi) * np.cos(theta)
    #     y = r * np.sin(phi) * np.sin(theta)
    #     z = 1.65 + r * np.cos(phi)
    #     points.append([x, y, z])
    
    # # 躯干（椭圆柱体）
    # for _ in range(num_points//3):
    #     theta = np.random.uniform(0, 2*np.pi)
    #     r = 0.15 + 0.05 * np.random.random()
    #     height = np.random.uniform(0.8, 1.6)
    #     x = r * np.cos(theta)
    #     y = 0.1 * r * np.sin(theta)  # 前后较窄
    #     z = height
    #     points.append([x, y, z])
    
    # # 手臂
    # for side in [-1, 1]:  # 左右手臂
    #     for _ in range(num_points//12):
    #         arm_pos = np.random.uniform(0, 0.6)  # 沿手臂长度
    #         x = side * (0.2 + arm_pos * 0.3)
    #         y = 0.05 * (np.random.random() - 0.5)
    #         z = 1.4 - arm_pos * 0.4
    #         points.append([x, y, z])
    
    # # 腿部
    # for side in [-0.08, 0.08]:  # 左右腿
    #     for _ in range(num_points//6):
    #         leg_height = np.random.uniform(0, 0.9)
    #         x = side + 0.03 * (np.random.random() - 0.5)
    #         y = 0.05 * (np.random.random() - 0.5)
    #         z = leg_height
    #         points.append([x, y, z])
    
    # # 填充剩余点
    # while len(points) < num_points:
    #     x = 0.7 * (np.random.random() - 0.5)
    #     y = 0.4 * (np.random.random() - 0.5)
    #     z = 1.8 * np.random.random()
    #     points.append([x, y, z])
    
    # return torch.tensor(points[:num_points], dtype=torch.float32)

def reconstruct_surface(  X: Union[torch.Tensor, np.ndarray], 
                          Y: Union[torch.Tensor, np.ndarray], 
                          Z: Union[torch.Tensor, np.ndarray], 
                          param: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从3D点云重建表面mesh
        
        Args:
            points: 点云数据，shape (N, 3)
            algorithm: 重建算法 ('normal_based', 'alpha_shape', 'ball_pivoting')
            param: 算法参数
                  
        Returns:
            triangles: 三角形索引，shape (M, 3)
            vertices: 顶点坐标，shape (N, 3)
        """
        # 输入验证和预处理
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        elif not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        elif not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)

        if isinstance(Z, np.ndarray):
            Z = torch.from_numpy(Z).float()
        elif not isinstance(Z, torch.Tensor):
            Z = torch.tensor(Z, dtype=torch.float32)

        try:
            # 确保数据在GPU上
            if not X.is_cuda:
                X = X.cuda()
            if not Y.is_cuda:
                Y = Y.cuda()
            if not Z.is_cuda:
                Z = Z.cuda()
            
            try:
                from pyTriangleMesh import triMesh
                cuda_module = triMesh
                _cuda_available = True
                print("CUDA扩展加载成功 (预编译)")
            except ImportError:
                pass


            if hasattr(cuda_module, 'surface_reconstruction') and hasattr(cuda_module, "find_four_grid_neighbors"):
                four_nn_inds = cuda_module.find_four_grid_neighbors(X, Y, Z)
                triangles_result = cuda_module.surface_reconstruction(X, Y, Z, four_nn_inds, param)
            else:
                raise AttributeError("CUDA模块中未找到表面重建函数")
            print("重建出的三角形 Tensor 的形状:", triangles_result.shape)
            print("重建出的三角形 Tensor 的数据类型:", triangles_result.dtype)
            print("重建出的三角形 Tensor 的设备:", triangles_result.device)

            vertices = torch.vstack([X.flatten().cpu(), Y.flatten().cpu(), Z.flatten().cpu()]).transpose(0,1)# 假设 points 是你在 Python 中传递给函数时使用的原始点云

            # 将三角形索引从 GPU 移回 CPU 并转换为 NumPy 数组
            triangles = triangles_result.cpu()
            return triangles, vertices
            
        except Exception as e:
            warnings.warn(f"CUDA重建失败，回退到CPU实现: {e}")

# 生成测试数据
print("生成行人点云...")
# pedestrian_points = generate_dense_pedestrian_pointcloud(1500)
x, y, z, np_points = generate_dense_pedestrian_pointcloud(1500)

# 测试不同的重建算法
algorithms = ['normal_based', 'alpha_shape', 'ball_pivoting']
params = [1.0, 1.5, 1.3]

results = {}


print(f"\n使用 Delaunay2.5D 算法重建表面...")


# mesh = reconstruct_surface(x,y,z, 0.5)
mesh = surface_reconstructor.build_pedestrian_surface(
    x,y,z, 
    algorithm='normal_based', 
    param=0.12,
    param2=0.1,
    smooth_iterations=2
)
results["Delaunay2.5D"] = mesh
print(f"  顶点数: {mesh['num_vertices']}")
print(f"  三角形数: {mesh['num_faces']}")
print(f"  平均面积: {mesh['quality_metrics']['mean_area']:.6f}")
print(f"  平均纵横比: {mesh['quality_metrics']['mean_aspect_ratio']:.3f}")

# for i, (algorithm, param) in enumerate(zip(algorithms, params)):
#     print(f"\n使用 {algorithm} 算法重建表面...")
    
#     mesh = surface_reconstructor.build_pedestrian_surface(
#         pedestrian_points, 
#         algorithm=algorithm, 
#         param=param,
#         smooth_iterations=2
#     )
    
#     results[algorithm] = mesh
    
#     print(f"  顶点数: {mesh['num_vertices']}")
#     print(f"  三角形数: {mesh['num_faces']}")
#     print(f"  平均面积: {mesh['quality_metrics']['mean_area']:.6f}")
#     print(f"  平均纵横比: {mesh['quality_metrics']['mean_aspect_ratio']:.3f}")

# 可视化比较
def visualize_surface_reconstruction(original_points, meshes, np_mode=False):
    """可视化表面重建结果"""
    fig = plt.figure(figsize=(20, 5))
    
    # 原始点云
    ax1 = fig.add_subplot(141, projection='3d')

        
    if isinstance(original_points, np.ndarray):
        points = original_points
    elif isinstance(original_points, torch.Tensor):
        points = original_points.numpy()
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=points[:, 2], cmap='viridis', s=2, alpha=0.6)
    ax1.set_title('原始点云')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # 三种算法结果
    titles = ['基于法向量', 'Alpha Shape', 'Ball Pivoting']
    for i, (algorithm, title) in enumerate(zip(algorithms, titles)):
        ax = fig.add_subplot(142 + i, projection='3d')
        
        mesh = meshes[algorithm]
        vertices = mesh['vertices'].numpy()
        triangles = mesh['faces'].numpy()
        
        # 绘制三角形（只显示部分以避免过于密集）
        for j, tri in enumerate(triangles[:min(200, len(triangles))]):
            triangle_points = vertices[tri]
            triangle_points = np.vstack([triangle_points, triangle_points[0]])
            ax.plot(triangle_points[:, 0], 
                   triangle_points[:, 1], 
                   triangle_points[:, 2], 'b-', alpha=0.3, linewidth=0.5)
        
        # 绘制顶点
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c=vertices[:, 2], cmap='plasma', s=1)
        
        ax.set_title(f'{title}\n面数: {mesh["num_faces"]}')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

# 可视化结果
# pedestrian_points = np_points.T
# visualize_surface_reconstruction(pedestrian_points, results, True)

# 保存最佳结果
def save_mesh_with_normals(mesh, filename):
    """保存包含法向量的OBJ文件"""
    vertices = mesh['vertices'].cpu().numpy()
    faces = mesh['faces'].cpu().numpy()
    normals = mesh['vertex_normals'].cpu().numpy()
    
    with open(filename, 'w') as f:
        # 写入顶点
        for i, v in enumerate(vertices):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # 写入法向量
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        # 写入面片（OBJ格式索引从1开始）
        for face in faces:
            f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
    
    print(f"带法向量的mesh已保存到: {filename}")

# # 选择最佳结果保存
# best_algorithm = 'normal_based'  # 通常效果较好
# best_mesh = results[best_algorithm]
# save_mesh_with_normals(best_mesh, "pedestrian_surface_mesh_0.obj")
# # 选择最佳结果保存
# best_algorithm = 'alpha_shape'  # 通常效果较好
# best_mesh = results[best_algorithm]
# save_mesh_with_normals(best_mesh, "pedestrian_surface_mesh_1.obj")
# # 选择最佳结果保存
# best_algorithm = 'ball_pivoting'  # 通常效果较好
# best_mesh = results[best_algorithm]
# save_mesh_with_normals(best_mesh, "pedestrian_surface_mesh_2.obj")
# # 选择最佳结果保存
best_algorithm = 'Delaunay2.5D'  # 通常效果较好
best_mesh = results[best_algorithm]
save_mesh_with_normals(best_mesh, "pedestrian_surface_mesh_delaunay.obj")
# 分析mesh质量
print(f"\n最佳mesh质量分析 ({best_algorithm}):")
quality = best_mesh['quality_metrics']
print(f"  总顶点数: {best_mesh['num_vertices']}")
print(f"  总面数: {best_mesh['num_faces']}")
print(f"  平均三角形面积: {quality['mean_area']:.6f}")
print(f"  平均纵横比: {quality['mean_aspect_ratio']:.3f}")
print(f"  面积范围: {quality['areas'].min():.6f} - {quality['areas'].max():.6f}")
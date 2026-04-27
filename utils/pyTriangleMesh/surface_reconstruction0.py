import torch
import numpy as np
import warnings
from typing import Tuple, Dict, Optional, Union

# import triMesh_cuda as triMesh

class SurfaceReconstruction:
    def __init__(self):
        """初始化表面重建器"""
        self.cuda_module = None
        self._cuda_available = False
        self._try_load_cuda_extension()
    
    def _try_load_cuda_extension(self):
        """尝试加载CUDA扩展"""
        try:
            # 方法1: 如果使用torch.utils.cpp_extension.load编译
            try:
                from torch.utils.cpp_extension import load
                import os
                
                # 获取当前文件的目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                cuda_file = os.path.join(project_root, 'pyTriangleMesh/src', 'surface_reconstruction.cu')
                
                if os.path.exists(cuda_file):
                    self.cuda_module = load(
                        name="surface_reconstruction_cuda",
                        sources=[cuda_file],
                        extra_cuda_cflags=["-O3", "--use_fast_math"],
                        verbose=False
                    )
                    self._cuda_available = True
                    print("CUDA扩展加载成功 (动态编译)")
                    return
            except Exception as e:
                print(f"动态编译CUDA扩展失败: {e}")
            
            # 方法2: 如果使用预编译的扩展
            try:
                from pyTriangleMesh import triMesh
                self.cuda_module = triMesh
                self._cuda_available = True
                print("CUDA扩展加载成功 (预编译)")
                return
            except ImportError:
                pass
            
            # 方法3: 直接导入（如果在当前目录）
            try:
                import surface_reconstruction_cuda
                self.cuda_module = surface_reconstruction_cuda
                self._cuda_available = True
                print("CUDA扩展加载成功 (直接导入)")
                return
            except ImportError:
                pass
            
        except Exception as e:
            warnings.warn(f"加载CUDA扩展失败: {e}")
        
        print("CUDA扩展不可用，将使用CPU备用实现")
    
    def is_cuda_available(self) -> bool:
        """检查CUDA扩展是否可用"""
        return self._cuda_available and torch.cuda.is_available()
    
    def reconstruct_surface(self, 
                          points: Union[torch.Tensor, np.ndarray], 
                          algorithm: str = 'ball_pivoting', 
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
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)

   
        
        if points.dim() != 2 or points.size(1) != 3:
            raise ValueError(f"点云数据形状应为 (N, 3)，当前为 {points.shape}")
        
        # 预处理点云
        points_clean = self._preprocess_points(points)
        
        if points_clean.size(0) < 3:
            raise ValueError("需要至少3个有效点来构建mesh")
        
        # 根据是否有CUDA扩展选择实现
        if self.is_cuda_available():
            return self._cuda_reconstruct(points_clean, algorithm, param)
        else:
            return self._cpu_reconstruct(points_clean, algorithm, param)
    
    def _cuda_reconstruct(self, points: torch.Tensor, algorithm: str, param: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用CUDA进行表面重建"""
        # 算法类型映射
        algorithm_map = {
            'normal_based': 0,
            'alpha_shape': 1,
            'ball_pivoting': 2
        }
        algorithm_type = algorithm_map.get(algorithm, 0)
        
        try:
            # 确保数据在GPU上
            if not points.is_cuda:
                points = points.cuda()
            
            # 检查CUDA扩展的可用函数
            if hasattr(self.cuda_module, 'surface_reconstruction'):
                # 使用简化版本
                triangles_result = self.cuda_module.surface_reconstruction(
                    points, algorithm_type, param
                )
            elif hasattr(self.cuda_module, 'surface_reconstruction_cuda'):
                # 使用完整版本 - 需要转换数据格式
                triangles_result = self._call_full_cuda_function(points, algorithm_type, param)
            else:
                raise AttributeError("CUDA模块中未找到表面重建函数")
            print("重建出的三角形 Tensor 的形状:", triangles_result.shape)
            print("重建出的三角形 Tensor 的数据类型:", triangles_result.dtype)
            print("重建出的三角形 Tensor 的设备:", triangles_result.device)
            vertices = points.cpu() # 假设 points 是你在 Python 中传递给函数时使用的原始点云

            # 将三角形索引从 GPU 移回 CPU 并转换为 NumPy 数组
            triangles = triangles_result.cpu()
            return triangles, vertices
            
        except Exception as e:
            warnings.warn(f"CUDA重建失败，回退到CPU实现: {e}")
            return self._cpu_reconstruct(points.cpu(), algorithm, param)
    
    def _call_full_cuda_function(self, points: torch.Tensor, algorithm_type: int, param: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """调用完整的CUDA函数（需要特殊的数据格式转换）"""
        num_points = points.size(0)
        max_triangles = min(2 * num_points, 20000)
        
        # 准备输出缓冲区
        triangles_buffer = torch.zeros((max_triangles * 3,), dtype=torch.int32, device='cpu')
        num_triangles = torch.zeros((1,), dtype=torch.int32, device='cpu')
        
        # 将点云数据转换为扁平格式
        points_flat = points.cpu().contiguous().view(-1)
        
        # 调用CUDA函数
        self.cuda_module.surface_reconstruction_cuda(
            points_flat.numpy().astype(np.float32),
            num_points,
            triangles_buffer.numpy(),
            num_triangles.numpy(),
            algorithm_type,
            float(param)
        )
        
        # 提取有效三角形
        valid_triangle_count = num_triangles.item()
        if valid_triangle_count > 0:
            triangles = triangles_buffer[:valid_triangle_count * 3].view(-1, 3)
        else:
            triangles = torch.empty((0, 3), dtype=torch.int32)
        
        return triangles, points
    
    def _cpu_reconstruct(self, points: torch.Tensor, algorithm: str, param: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU备用实现"""
        print(f"使用CPU备用实现 ({algorithm})")
        
        num_points = points.size(0)
        
        if algorithm == 'alpha_shape':
            return self._cpu_alpha_shape(points, param)
        elif algorithm == 'ball_pivoting':
            return self._cpu_ball_pivoting(points, param)
        else:  # normal_based 或其他
            return self._cpu_delaunay(points)
    
    def _cpu_delaunay(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU Delaunay三角化"""
        try:
            from scipy.spatial import Delaunay
            
            # 使用前两个维度进行2D Delaunay三角化
            points_2d = points[:, :2].cpu().numpy()
            tri = Delaunay(points_2d)
            triangles = torch.from_numpy(tri.simplices).long()
            
            return triangles, points
            
        except ImportError:
            # 如果没有scipy，使用简单的扇形三角化
            return self._simple_fan_triangulation(points)
    
    def _cpu_alpha_shape(self, points: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU Alpha Shape实现（简化版）"""
        try:
            from scipy.spatial import Delaunay
            
            points_np = points.cpu().numpy()
            tri = Delaunay(points_np[:, :2])
            
            # 简化的alpha shape：过滤掉外接圆半径大于alpha的三角形
            valid_triangles = []
            for simplex in tri.simplices:
                # 计算三角形的外接圆半径
                p1, p2, p3 = points_np[simplex]
                
                # 计算边长
                a = np.linalg.norm(p2 - p3)
                b = np.linalg.norm(p1 - p3)
                c = np.linalg.norm(p1 - p2)
                
                # 计算面积
                s = (a + b + c) / 2
                area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
                
                # 计算外接圆半径
                if area > 1e-10:
                    circumradius = (a * b * c) / (4 * area)
                    if circumradius < alpha:
                        valid_triangles.append(simplex)
            
            if valid_triangles:
                triangles = torch.from_numpy(np.array(valid_triangles)).long()
            else:
                triangles = torch.empty((0, 3), dtype=torch.long)
            
            return triangles, points
            
        except ImportError:
            return self._simple_fan_triangulation(points)
    
    def _cpu_ball_pivoting(self, points: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU Ball Pivoting实现（极简版）"""
        # 这里使用简化的近邻连接方法
        return self._simple_neighbor_triangulation(points, radius)
    
    def _simple_fan_triangulation(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """简单的扇形三角化"""
        num_points = points.size(0)
        if num_points < 3:
            return torch.empty((0, 3), dtype=torch.long), points
        
        triangles = []
        # 以第一个点为中心创建扇形三角形
        for i in range(1, num_points - 1):
            triangles.append([0, i, i + 1])
        
        if triangles:
            triangles_tensor = torch.tensor(triangles, dtype=torch.long)
        else:
            triangles_tensor = torch.empty((0, 3), dtype=torch.long)
        
        return triangles_tensor, points
    
    def _simple_neighbor_triangulation(self, points: torch.Tensor, max_dist: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于近邻距离的简单三角化"""
        num_points = points.size(0)
        if num_points < 3:
            return torch.empty((0, 3), dtype=torch.long), points
        
        triangles = []
        points_np = points.cpu().numpy()
        
        # 对每个点找近邻形成三角形
        for i in range(num_points):
            distances = np.linalg.norm(points_np - points_np[i], axis=1)
            neighbors = np.where((distances > 0) & (distances < max_dist))[0]
            
            # 在近邻中形成三角形
            for j in range(len(neighbors) - 1):
                for k in range(j + 1, len(neighbors)):
                    if len(neighbors) >= 2:
                        triangles.append([i, neighbors[j], neighbors[k]])
        
        if triangles:
            # 去重
            triangles = list(set(tuple(sorted(tri)) for tri in triangles))
            triangles_tensor = torch.tensor(triangles, dtype=torch.long)
        else:
            triangles_tensor = torch.empty((0, 3), dtype=torch.long)
        
        return triangles_tensor, points
    
    def _preprocess_points(self, points: torch.Tensor) -> torch.Tensor:
        """预处理点云"""
        # 移除NaN和无穷大值
        finite_mask = torch.isfinite(points).all(dim=1)
        points_clean = points[finite_mask]
        
        if points_clean.size(0) == 0:
            raise ValueError("所有点都包含NaN或无穷大值")
        
        # 移除重复点（使用较大的容差）
        # if points_clean.size(0) > 1:
        #     # 简单的重复点移除
        #     unique_points = []
        #     for i, point in enumerate(points_clean):
        #         is_duplicate = False
        #         for existing_point in unique_points:
        #             if torch.norm(point - existing_point) < 1e-6:
        #                 is_duplicate = True
        #                 break
        #         if not is_duplicate:
        #             unique_points.append(point)
            
        #     if unique_points:
        #         points_clean = torch.stack(unique_points)
        
        return points_clean


class PedestrianSurfaceReconstructor:
    """专门针对行人点云的表面重建器"""
    
    def __init__(self):
        self.reconstructor = SurfaceReconstruction()
    
    def build_pedestrian_surface(self, 
                                lidar_points: Union[torch.Tensor, np.ndarray], 
                                algorithm: str = 'ball_pivoting', 
                                param: float = 0.1, 
                                smooth_iterations: int = 2) -> Dict:
        """
        为行人点云构建表面mesh
        
        Args:
            lidar_points: 激光雷达点云 (N, 3)
            algorithm: 重建算法
            param: 算法参数
            smooth_iterations: 平滑迭代次数
            
        Returns:
            mesh: 包含顶点、面片、法向量等的字典
        """
        try:
            # 表面重建
            triangles, vertices = self.reconstructor.reconstruct_surface(
                lidar_points, algorithm, param
            )
         
            # 移除退化三角形
            triangles_clean = self._remove_degenerate_triangles(vertices, triangles)
            
            # 平滑处理
            if smooth_iterations > 0 and triangles_clean.size(0) > 0:
                vertices_smooth = self._smooth_mesh(vertices, triangles_clean, smooth_iterations)
            else:
                vertices_smooth = vertices
            
            # 计算质量指标
            quality_metrics = self._compute_quality_metrics(vertices_smooth, triangles_clean)
            
            # 计算法向量（如果有三角形）
            if triangles_clean.size(0) > 0:
                vertex_normals = self._compute_vertex_normals(vertices_smooth, triangles_clean)
            else:
                vertex_normals = torch.zeros_like(vertices_smooth)
            
            return {
                'vertices': vertices_smooth,
                'faces': triangles_clean,
                'vertex_normals': vertex_normals,
                'quality_metrics': quality_metrics,
                'num_vertices': vertices_smooth.size(0),
                'num_faces': triangles_clean.size(0),
                'algorithm_used': algorithm
            }
            
        except Exception as e:
            print(f"表面重建失败: {e}")
            # 返回空mesh
            return {
                'vertices': torch.empty((0, 3)),
                'faces': torch.empty((0, 3), dtype=torch.long),
                'vertex_normals': torch.empty((0, 3)),
                'quality_metrics': {'mean_area': 0.0, 'mean_aspect_ratio': 0.0},
                'num_vertices': 0,
                'num_faces': 0,
                'algorithm_used': algorithm,
                'error': str(e)
            }
    
    def _remove_degenerate_triangles(self, vertices: torch.Tensor, triangles: torch.Tensor, min_area: float = 1e-6) -> torch.Tensor:
        """移除退化三角形"""
        if triangles.size(0) == 0:
            return triangles
        
        # 确保索引在有效范围内
        valid_indices = (triangles >= 0) & (triangles < vertices.size(0))
        valid_mask = valid_indices.all(dim=1)
        triangles = triangles[valid_mask]
        
        if triangles.size(0) == 0:
            return triangles
        
        # 计算三角形面积
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
        areas = 0.5 * torch.norm(cross_product, dim=1)
        
        # 保留面积大于阈值的三角形
        valid_mask = areas > min_area
        return triangles[valid_mask]
    
    def _smooth_mesh(self, vertices: torch.Tensor, triangles: torch.Tensor, iterations: int = 3, lambda_param: float = 0.1) -> torch.Tensor:
        """简化的拉普拉斯平滑"""
        if triangles.size(0) == 0:
            return vertices
        
        smoothed_vertices = vertices.clone()
        
        for _ in range(iterations):
            new_vertices = smoothed_vertices.clone()
            
            # 简化版：对每个顶点，计算其在三角形中的邻居平均位置
            for i in range(vertices.size(0)):
                # 找到包含顶点i的所有三角形
                mask = (triangles == i).any(dim=1)
                if not mask.any():
                    continue
                
                connected_triangles = triangles[mask]
                neighbors = set()
                
                for tri in connected_triangles:
                    for v in tri:
                        if v.item() != i:
                            neighbors.add(v.item())
                
                if neighbors:
                    neighbor_positions = smoothed_vertices[list(neighbors)]
                    centroid = torch.mean(neighbor_positions, dim=0)
                    new_vertices[i] = (1 - lambda_param) * smoothed_vertices[i] + lambda_param * centroid
            
            smoothed_vertices = new_vertices
        
        return smoothed_vertices
    
    def _compute_quality_metrics(self, vertices: torch.Tensor, triangles: torch.Tensor) -> Dict:
        """计算mesh质量指标"""
        if triangles.size(0) == 0:
            return {'mean_area': 0.0, 'mean_aspect_ratio': 0.0, 'areas': torch.empty(0)}
        
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        # 边长
        edge1 = torch.norm(v1 - v0, dim=1)
        edge2 = torch.norm(v2 - v1, dim=1)
        edge3 = torch.norm(v0 - v2, dim=1)
        
        # 面积
        cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
        areas = 0.5 * torch.norm(cross_product, dim=1)
        
        # 纵横比
        min_edges = torch.min(torch.min(edge1, edge2), edge3)
        max_edges = torch.max(torch.max(edge1, edge2), edge3)
        aspect_ratios = max_edges / (min_edges + 1e-8)
        
        return {
            'areas': areas,
            'aspect_ratios': aspect_ratios,
            'mean_area': torch.mean(areas).item(),
            'mean_aspect_ratio': torch.mean(aspect_ratios).item()
        }
    
    def _compute_vertex_normals(self, vertices: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
        """计算顶点法向量"""
        if triangles.size(0) == 0:
            return torch.zeros_like(vertices)
        
        vertex_normals = torch.zeros_like(vertices)
        
        # 计算每个三角形的法向量
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        
        # 将面法向量累加到顶点
        for i, tri in enumerate(triangles):
            for j in range(3):
                vertex_idx = tri[j].item()
                if vertex_idx < vertex_normals.size(0):
                    vertex_normals[vertex_idx] += face_normals[i]
        
        # 归一化顶点法向量
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)
        
        return vertex_normals
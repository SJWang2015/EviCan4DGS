#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>     // For sqrtf
#include <stdio.h>    // For printf (debugging)
#include <float.h>    // For FLT_MAX

// 定义常量
#define MAX_TRIANGLES 2000000 // 根据你的需求调整，需要足够大以容纳所有可能的三角形
#define EPSILON 1e-6          // 小的浮点数容差，用于避免除以零和处理退化情况

// Point3D 结构体定义 (与之前相同)
struct Point3D {
    float x, y, z;
    __device__ __host__ Point3D() : x(0), y(0), z(0) {}
    __device__ __host__ Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    __device__ __host__ Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }
    
    __device__ __host__ Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.z);
    }
    
    __device__ __host__ Point3D operator*(float scalar) const {
        return Point3D(x * scalar, y * scalar, z * scalar);
    }
    
    __device__ __host__ float dot(const Point3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    __device__ __host__ Point3D cross(const Point3D& other) const {
        return Point3D(y * other.z - z * other.y,
                      z * other.x - x * other.z,
                      x * other.y - y * other.x);
    }
    
    __device__ __host__ float norm() const {
        return sqrtf(x * x + y * y + z * z);
    }
    
    __device__ __host__ Point3D normalize() const {
        float n = norm();
        return n > 1e-6 ? Point3D(x/n, y/n, z/n) : Point3D(0, 0, 0); 
    }
};

// Triangle 结构体定义
struct Triangle {
    int v0, v1, v2;
    __device__ __host__ Triangle() : v0(-1), v1(-1), v2(-1) {}
    __device__ __host__ Triangle(int a, int b, int c) : v0(a), v1(b), v2(c) {}
};

// 计算两点之间距离的辅助函数 (Device function)
__device__ float calculate_distance(const Point3D& p1, const Point3D& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

__device__ float calculate_norm_consistence(const Point3D& p1, const Point3D& p2, const Point3D& center, const Point3D& center_normal){
    Point3D edge1 = p1 - center;
    Point3D edge2 = p2 - center;
    Point3D tri_normal = edge1.cross(edge2);

    float tri_normal_norm = tri_normal.norm();
    tri_normal = tri_normal * (1.0f / tri_normal_norm);

    float normal_consistency = tri_normal.dot(center_normal);

    return std::abs(normal_consistency);
}


// // CUDA 核函数：为每个非零点搜索其沿上下左右四个方向的最近非零邻居
// // output_neighbors 数组的布局为：对于每个点 current_idx，存储
// // output_neighbors[current_idx * 4 + 0] = neighbor_up_idx
// // output_neighbors[current_idx * 4 + 1] = neighbor_down_idx
// // output_neighbors[current_idx * 4 + 2] = neighbor_left_idx
// // output_neighbors[current_idx * 4 + 3] = neighbor_right_idx
// __global__ void findFourGridNeighborsKernel(
//     const float *__restrict__ Xw,      // 输入：展平的 X 坐标 (大小为 iRows * iCols)
//     const float *__restrict__ Yw,      // 输入：展平的 Y 坐标 (大小为 iRows * iCols)
//     const float *__restrict__ Zw,      // 输入：展平的 Z 坐标 (大小为 iRows * iCols)
//     int iRows,                         // 输入：原始网格的行数
//     int iCols,                         // 输入：原始网格的列数
//     int *__restrict__ output_neighbors) // 输出：存储四个方向邻居索引 (iRows * iCols * 4)
// {
//     // 每个线程处理一个网格单元 (row, col)
//     int col = blockIdx.x * blockDim.x + threadIdx.x; // 当前列索引
//     int row = blockIdx.y * blockDim.y + threadIdx.y; // 当前行索引

//     // 确保线程在有效网格范围内
//     if (row >= iRows || col >= iCols) return;

//     int current_idx = row * iCols + col; // 当前点的线性索引
//     int output_offset = current_idx * 4;

//     // 初始化所有邻居为无效 (-1)
//     output_neighbors[output_offset + 0] = -1; // Up
//     output_neighbors[output_offset + 1] = -1; // Down
//     output_neighbors[output_offset + 2] = -1; // Left
//     output_neighbors[output_offset + 3] = -1; // Right

//     // 如果当前点是无效点 (Z = 0.0)，则跳过
//     if (Zw[current_idx] == 0.0f) {
//         return;
//     }

//     // ====== 搜索向上邻居 ======
//     for (int r = row - 1; r >= 0; --r) {
//         int neighbor_idx = r * iCols + col;
//         if (Zw[neighbor_idx] != 0.0f) {
//             output_neighbors[output_offset + 0] = neighbor_idx;
//             break; // 找到第一个非零点后停止搜索
//         }
//     }

//     // ====== 搜索向下邻居 ======
//     for (int r = row + 1; r < iRows; ++r) {
//         int neighbor_idx = r * iCols + col;
//         if (Zw[neighbor_idx] != 0.0f) {
//             output_neighbors[output_offset + 1] = neighbor_idx;
//             break; // 找到第一个非零点后停止搜索
//         }
//     }

//     // ====== 搜索向左邻居 ======
//     for (int c = col - 1; c >= 0; --c) {
//         int neighbor_idx = row * iCols + c;
//         if (Zw[neighbor_idx] != 0.0f) {
//             output_neighbors[output_offset + 2] = neighbor_idx;
//             break; // 找到第一个非零点后停止搜索
//         }
//     }

//     // ====== 搜索向右邻居 ======
//     for (int c = col + 1; c < iCols; ++c) {
//         int neighbor_idx = row * iCols + c;
//         if (Zw[neighbor_idx] != 0.0f) {
//             output_neighbors[output_offset + 3] = neighbor_idx;
//             break; // 找到第一个非零点后停止搜索
//         }
//     }
// }

// CUDA 核函数：为每个非零点搜索其沿上下左右四个方向的最近非零邻居
// output_neighbors 数组的布局为：对于每个点 current_idx，存储
// output_neighbors[current_idx * 4 + 0] = neighbor_up_idx
// output_neighbors[current_idx * 4 + 1] = neighbor_down_idx
// output_neighbors[current_idx * 4 + 2] = neighbor_left_idx
// output_neighbors[current_idx * 4 + 3] = neighbor_right_idx
__global__ void findFourGridNeighborsKernel(
    const float *__restrict__ Xw,      // 输入：展平的 X 坐标 (大小为 iRows * iCols)
    const float *__restrict__ Yw,      // 输入：展平的 Y 坐标 (大小为 iRows * iCols)
    const float *__restrict__ Zw,      // 输入：展平的 Z 坐标 (大小为 iRows * iCols)
    int iRows,                         // 输入：原始网格的行数
    int iCols,                         // 输入：原始网格的列数
    int *__restrict__ output_neighbors) // 输出：存储四个方向邻居索引 (iRows * iCols * 4)
{
    // 每个线程处理一个网格单元 (row, col)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 当前列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 当前行索引

    // 确保线程在有效网格范围内
    if (row >= iRows || col >= iCols) return;

    int current_idx = row * iCols + col; // 当前点的线性索引
    int output_offset = current_idx * 4;

    // 初始化所有邻居为无效 (-1)
    output_neighbors[output_offset + 0] = -1; // Up
    output_neighbors[output_offset + 1] = -1; // Down
    output_neighbors[output_offset + 2] = -1; // Left
    output_neighbors[output_offset + 3] = -1; // Right

    // 如果当前点是无效点 (Z = 0.0)，则跳过
    if (Zw[current_idx] == 0.0f) {
        return;
    }

    Point3D center_pt = Point3D(Xw[current_idx], Yw[current_idx], Zw[current_idx]);

    // ====== 搜索向上邻居 (在上方所有行中查找最近的非零点) ======
    // 注意: 这种全局搜索模式对于每个线程都非常耗时，可能导致性能瓶颈。
    // 如果网格很大，请考虑优化策略或更稀疏的邻居定义。
    float min_dist_up = FLT_MAX;
    int best_up_idx = -1;
    for (int r_search = row - 1; r_search >= 0; --r_search) { // 遍历当前行上方的所有行
        for (int c_search = 0; c_search < iCols; ++c_search) { // 遍历该行中的所有列
            int neighbor_idx = r_search * iCols + c_search;
            if (Zw[neighbor_idx] != 0.0f) {
                Point3D neighbor_pt = Point3D(Xw[neighbor_idx], Yw[neighbor_idx], Zw[neighbor_idx]);
                float dist = calculate_distance(center_pt, neighbor_pt);
                if (dist < min_dist_up) {
                    min_dist_up = dist;
                    best_up_idx = neighbor_idx;
                }
            }
        }
    }
    output_neighbors[output_offset + 0] = best_up_idx;

    // ====== 搜索向下邻居 (在下方所有行中查找最近的非零点) ======
    float min_dist_down = FLT_MAX;
    int best_down_idx = -1;
    for (int r_search = row + 1; r_search < iRows; ++r_search) { // 遍历当前行下方的所有行
        for (int c_search = 0; c_search < iCols; ++c_search) { // 遍历该行中的所有列
            int neighbor_idx = r_search * iCols + c_search;
            if (Zw[neighbor_idx] != 0.0f) {
                Point3D neighbor_pt = Point3D(Xw[neighbor_idx], Yw[neighbor_idx], Zw[neighbor_idx]);
                float dist = calculate_distance(center_pt, neighbor_pt);
                if (dist < min_dist_down) {
                    min_dist_down = dist;
                    best_down_idx = neighbor_idx;
                }
            }
        }
    }
    output_neighbors[output_offset + 1] = best_down_idx;

    // ====== 搜索向左邻居 (在左方所有列中查找最近的非零点) ======
    float min_dist_left = FLT_MAX;
    int best_left_idx = -1;
    for (int c_search = col - 1; c_search >= 0; --c_search) { // 遍历当前列左方的所有列
        for (int r_search = 0; r_search < iRows; ++r_search) { // 遍历该列中的所有行
            int neighbor_idx = r_search * iCols + c_search;
            if (Zw[neighbor_idx] != 0.0f) {
                Point3D neighbor_pt = Point3D(Xw[neighbor_idx], Yw[neighbor_idx], Zw[neighbor_idx]);
                float dist = calculate_distance(center_pt, neighbor_pt);
                if (dist < min_dist_left) {
                    min_dist_left = dist;
                    best_left_idx = neighbor_idx;
                }
            }
        }
    }
    output_neighbors[output_offset + 2] = best_left_idx;

    // ====== 搜索向右邻居 (在右方所有列中查找最近的非零点) ======
    float min_dist_right = FLT_MAX;
    int best_right_idx = -1;
    for (int c_search = col + 1; c_search < iCols; ++c_search) { // 遍历当前列右方的所有列
        for (int r_search = 0; r_search < iRows; ++r_search) { // 遍历该列中的所有行
            int neighbor_idx = r_search * iCols + c_search;
            if (Zw[neighbor_idx] != 0.0f) {
                Point3D neighbor_pt = Point3D(Xw[neighbor_idx], Yw[neighbor_idx], Zw[neighbor_idx]);
                float dist = calculate_distance(center_pt, neighbor_pt);
                if (dist < min_dist_right) {
                    min_dist_right = dist;
                    best_right_idx = neighbor_idx;
                }
            }
        }
    }
    output_neighbors[output_offset + 3] = best_right_idx;
}

// 新增的 CUDA 核函数：根据邻居计算点法向量
__global__ void calculateNormalsKernel(
    const float *__restrict__ Xw,      // 输入：展平的 X 坐标
    const float *__restrict__ Yw,      // 输入：展平的 Y 坐标
    const float *__restrict__ Zw,      // 输入：展平的 Z 坐标
    int iRows,                         // 输入：原始网格的行数
    int iCols,                         // 输入：原始网格的列数
    const int *__restrict__ four_neighbors, // 输入：每个点的四个方向邻居索引 (iRows * iCols * 4)
    float *__restrict__ normals_out)    // 输出：存储每个点法向量的数组 (iRows * iCols * 3)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= iRows || col >= iCols) return;

    int center_idx = row * iCols + col;
    int normals_offset = center_idx * 3; // 法向量输出的起始偏移

    // 初始化法向量为 (0, 0, 0)
    normals_out[normals_offset + 0] = 0.0f;
    normals_out[normals_offset + 1] = 0.0f;
    normals_out[normals_offset + 2] = 0.0f;

    // 如果中心点是无效点，则不计算法向量
    if (Zw[center_idx] == 0.0f) {
        return;
    }

    Point3D center_pt = Point3D(Xw[center_idx], Yw[center_idx], Zw[center_idx]);

    int neighbors_offset = center_idx * 4;
    int up_idx    = four_neighbors[neighbors_offset + 0];
    int down_idx  = four_neighbors[neighbors_offset + 1];
    int left_idx  = four_neighbors[neighbors_offset + 2];
    int right_idx = four_neighbors[neighbors_offset + 3];

    // 存储潜在的邻居点
    Point3D up_pt, down_pt, left_pt, right_pt;
    bool has_up = (up_idx != -1 && Zw[up_idx] != 0.0f);
    if (has_up) up_pt = Point3D(Xw[up_idx], Yw[up_idx], Zw[up_idx]);

    bool has_down = (down_idx != -1 && Zw[down_idx] != 0.0f);
    if (has_down) down_pt = Point3D(Xw[down_idx], Yw[down_idx], Zw[down_idx]);

    bool has_left = (left_idx != -1 && Zw[left_idx] != 0.0f);
    if (has_left) left_pt = Point3D(Xw[left_idx], Yw[left_idx], Zw[left_idx]);

    bool has_right = (right_idx != -1 && Zw[right_idx] != 0.0f);
    if (has_right) right_pt = Point3D(Xw[right_idx], Yw[right_idx], Zw[right_idx]);

    Point3D sum_normals = Point3D(0.0f, 0.0f, 0.0f);
    int num_contributing_triangles = 0;

    // Helper to calculate and add normalized triangle normal
    auto add_triangle_normal = [&](const Point3D& p1, const Point3D& p2, const Point3D& p3) {
        Point3D vec1 = p2 - p1;
        Point3D vec2 = p3 - p1;
        Point3D cross_prod = vec1.cross(vec2);
        float n_norm = cross_prod.norm();
        if (n_norm > EPSILON) { // 避免退化三角形
            Point3D normal = cross_prod * (1.0f / n_norm);
            // 确保法向量的 Z 分量大致朝向观察者（通常为正）
            // 这取决于你的坐标系和期望的表面方向。这里假设Z朝外。
            if (normal.z < 0.0f) {
                normal = normal * -1.0f; // 反转方向
            }
            sum_normals = sum_normals + normal;
            num_contributing_triangles++;
        }
    };

    // Calculate normals for up to four triangles around the center point
    // (Center, Right_Neighbor, Down_Neighbor)
    if (has_right && has_down) {
        add_triangle_normal(center_pt, right_pt, down_pt);
    }

    // (Center, Down_Neighbor, Left_Neighbor)
    if (has_down && has_left) {
        add_triangle_normal(center_pt, down_pt, left_pt);
    }

    // (Center, Left_Neighbor, Up_Neighbor)
    if (has_left && has_up) {
        add_triangle_normal(center_pt, left_pt, up_pt);
    }

    // (Center, Up_Neighbor, Right_Neighbor)
    if (has_up && has_right) {
        add_triangle_normal(center_pt, up_pt, right_pt);
    }

    if (num_contributing_triangles > 0) {
        Point3D final_normal = sum_normals.normalize();
        normals_out[normals_offset + 0] = final_normal.x;
        normals_out[normals_offset + 1] = final_normal.y;
        normals_out[normals_offset + 2] = final_normal.z;
    }
    // else: 法向量保持 (0,0,0) (初始化值)
}



// CUDA 核函数：根据中心点和其四个方向邻居构造三角形
__global__ void createTrianglesFromNeighborsKernel(
    const float *__restrict__ Xw,      // 输入：展平的 X 坐标
    const float *__restrict__ Yw,      // 输入：展平的 Y 坐标
    const float *__restrict__ Zw,      // 输入：展平的 Z 坐标
    int iRows,                         // 输入：原始网格的行数
    int iCols,                         // 输入：原始网格的列数
    const int *__restrict__ four_neighbors, // 输入：每个点的四个方向邻居索引 (iRows * iCols * 4)
    const float *__restrict__ normals,      // 输入：展平的 normals 
    Triangle *__restrict__ triangles_out,   // 输出：存储 Triangle 结构体的数组
    int *__restrict__ num_triangles_out,    // 输出：总三角形数量的原子计数器
    float dMaxEdgeLengthThreshold,           // 输入：最大边长阈值
    float dMaxNormalConststencyThreshold)           // 输入：最大边长阈值
{
    // 每个线程处理一个网格单元 (row, col)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= iRows || col >= iCols) return;

    int center_idx = row * iCols + col;
    // 如果中心点是无效点，则不生成三角形
    if (Zw[center_idx] == 0.0f) {
        return;
    }

    Point3D center_pt = Point3D(Xw[center_idx], Yw[center_idx], Zw[center_idx]);

    int normals_offset = center_idx * 3; // 计算当前点在 normals_out 数组中的起始偏移
    Point3D center_norm = Point3D(normals[normals_offset+0], normals[normals_offset+1], normals[normals_offset+2]);

    int neighbors_offset = center_idx * 4;
    int up_idx    = four_neighbors[neighbors_offset + 0];
    int down_idx  = four_neighbors[neighbors_offset + 1];
    int left_idx  = four_neighbors[neighbors_offset + 2];
    int right_idx = four_neighbors[neighbors_offset + 3];

    // 存储潜在的邻居点，如果索引为 -1，则表示无效
    Point3D up_pt, down_pt, left_pt, right_pt;
    bool has_up = (up_idx != -1 && Zw[up_idx] != 0.0f);
    if (has_up) up_pt = Point3D(Xw[up_idx], Yw[up_idx], Zw[up_idx]);

    bool has_down = (down_idx != -1 && Zw[down_idx] != 0.0f);
    if (has_down) down_pt = Point3D(Xw[down_idx], Yw[down_idx], Zw[down_idx]);

    bool has_left = (left_idx != -1 && Zw[left_idx] != 0.0f);
    if (has_left) left_pt = Point3D(Xw[left_idx], Yw[left_idx], Zw[left_idx]);

    bool has_right = (right_idx != -1 && Zw[right_idx] != 0.0f);
    if (has_right) right_pt = Point3D(Xw[right_idx], Yw[right_idx], Zw[right_idx]);

    // 尝试构造四个三角形，并进行边长检查
    // 顶点顺序 Center -> 邻居1 -> 邻居2 (例如：逆时针)

    // 1. Triangle (Center, Right_Neighbor, Down_Neighbor)
    if (has_right && has_down) {
        float e1 = calculate_distance(center_pt, right_pt);
        float e2 = calculate_distance(right_pt, down_pt);
        float e3 = calculate_distance(down_pt, center_pt);
        float norm_consist = calculate_norm_consistence(right_pt, down_pt, center_pt, center_norm);
        // if ((e1 <= dMaxEdgeLengthThreshold && e2 <= dMaxEdgeLengthThreshold && e3 <= dMaxEdgeLengthThreshold)|| (e1+e2+e3 <=dMaxEdgeLengthThreshold*5.0)) {
        if (e1 <= dMaxEdgeLengthThreshold && e2 <= dMaxEdgeLengthThreshold && e3 <= dMaxEdgeLengthThreshold && norm_consist >=dMaxNormalConststencyThreshold) {
            int tri_idx = atomicAdd(num_triangles_out, 1);
            if (tri_idx < MAX_TRIANGLES) {
                triangles_out[tri_idx] = Triangle(center_idx, right_idx, down_idx);
            }
        }
    }

    // 2. Triangle (Center, Down_Neighbor, Left_Neighbor)
    if (has_down && has_left) {
        float e1 = calculate_distance(center_pt, down_pt);
        float e2 = calculate_distance(down_pt, left_pt);
        float e3 = calculate_distance(left_pt, center_pt);
        float norm_consist = calculate_norm_consistence(down_pt, left_pt, center_pt, center_norm);
        // if ((e1 <= dMaxEdgeLengthThreshold && e2 <= dMaxEdgeLengthThreshold && e3 <= dMaxEdgeLengthThreshold)|| (e1+e2+e3 <=dMaxEdgeLengthThreshold*5.0)) {
        if (e1 <= dMaxEdgeLengthThreshold && e2 <= dMaxEdgeLengthThreshold && e3 <= dMaxEdgeLengthThreshold && norm_consist >=dMaxNormalConststencyThreshold) {
            int tri_idx = atomicAdd(num_triangles_out, 1);
            if (tri_idx < MAX_TRIANGLES) {
                triangles_out[tri_idx] = Triangle(center_idx, down_idx, left_idx);
            }
        }
    }

    // 3. Triangle (Center, Left_Neighbor, Up_Neighbor)
    if (has_left && has_up) {
        float e1 = calculate_distance(center_pt, left_pt);
        float e2 = calculate_distance(left_pt, up_pt);
        float e3 = calculate_distance(up_pt, center_pt);
        float norm_consist = calculate_norm_consistence(left_pt, up_pt, center_pt, center_norm);
        // if ((e1 <= dMaxEdgeLengthThreshold && e2 <= dMaxEdgeLengthThreshold && e3 <= dMaxEdgeLengthThreshold)|| (e1+e2+e3 <=dMaxEdgeLengthThreshold*5.0)) {
        if (e1 <= dMaxEdgeLengthThreshold && e2 <= dMaxEdgeLengthThreshold && e3 <= dMaxEdgeLengthThreshold && norm_consist >=dMaxNormalConststencyThreshold) {
            int tri_idx = atomicAdd(num_triangles_out, 1);
            if (tri_idx < MAX_TRIANGLES) {
                triangles_out[tri_idx] = Triangle(center_idx, left_idx, up_idx);
            }
        }
    }

    // 4. Triangle (Center, Up_Neighbor, Right_Neighbor)
    if (has_up && has_right) {
        float e1 = calculate_distance(center_pt, up_pt);
        float e2 = calculate_distance(up_pt, right_pt);
        float e3 = calculate_distance(right_pt, center_pt);
        float norm_consist = calculate_norm_consistence(up_pt, right_pt, center_pt, center_norm);
        // if ((e1 <= dMaxEdgeLengthThreshold && e2 <= dMaxEdgeLengthThreshold && e3 <= dMaxEdgeLengthThreshold)|| (e1+e2+e3 <=dMaxEdgeLengthThreshold*5.0)) {
        if (e1 <= dMaxEdgeLengthThreshold && e2 <= dMaxEdgeLengthThreshold && e3 <= dMaxEdgeLengthThreshold && norm_consist >=dMaxNormalConststencyThreshold) {
            int tri_idx = atomicAdd(num_triangles_out, 1);
            if (tri_idx < MAX_TRIANGLES) {
                triangles_out[tri_idx] = Triangle(center_idx, up_idx, right_idx);
            }
        }
    }
}


// 主机端封装函数，用于启动 CUDA 核函数
extern "C" {
    // Stage 1: 查找每个点的四个方向邻居
    void find_four_grid_neighbors_cuda_wrapper(
        const float* Xw, const float* Yw, const float* Zw, // 输入：展平的 X, Y, Z 坐标
        int iRows, int iCols,                              // 输入：网格维度
        int* output_neighbors,                             // 输出：存储四个方向邻居索引 (iRows * iCols * 4)
        cudaStream_t stream)                               // 输入：CUDA 流
    {
        dim3 blockSize(16, 16);
        dim3 gridSize(
            (iCols + blockSize.x - 1) / blockSize.x,
            (iRows + blockSize.y - 1) / blockSize.y
        );
        
        if (iRows <= 0 || iCols <= 0) return;

        findFourGridNeighborsKernel<<<gridSize, blockSize, 0, stream>>>(
            Xw, Yw, Zw,
            iRows, iCols,
            output_neighbors
        );
        cudaDeviceSynchronize();
    }

    // Stage 2: 计算法向量
    void calculate_normals_cuda_wrapper(
        const float* Xw, const float* Yw, const float* Zw, // 输入：展平的 X, Y, Z 坐标
        int iRows, int iCols,                              // 输入：网格维度
        const int* four_neighbors,                         // 输入：四个方向邻居索引
        float* normals_out,                                // 输出：每个点法向量的数组 (iRows * iCols * 3)
        cudaStream_t stream)                               // 输入：CUDA 流
    {
        dim3 blockSize(16, 16);
        dim3 gridSize(
            (iCols + blockSize.x - 1) / blockSize.x,
            (iRows + blockSize.y - 1) / blockSize.y
        );

        if (iRows <= 0 || iCols <= 0) return;

        calculateNormalsKernel<<<gridSize, blockSize, 0, stream>>>(
            Xw, Yw, Zw,
            iRows, iCols,
            four_neighbors,
            normals_out
        );
        cudaDeviceSynchronize();
    }

    // Stage 3: 根据邻居信息构造三角形
    void create_triangles_from_neighbors_cuda_wrapper(
        const float* Xw, const float* Yw, const float* Zw, // 输入：展平的 X, Y, Z 坐标
        int iRows, int iCols,                              // 输入：网格维度
        const int* four_neighbors,                         // 输入：四个方向邻居索引
        const float* normals,
        int* triangles_out,                                // 输出：GPU 指针，用于存储三角形索引
        int* num_triangles_out,                            // 输出：GPU 指针，用于存储生成的三角形数量
        float dMaxEdgeLengthThreshold,                     // 输入：最大边长阈值
        float dMaxNormalConststencyThreshold,
        cudaStream_t stream)                               // 输入：CUDA 流
    {
        // 将 GPU 上的三角形计数器初始化为零
        int zero = 0;
        cudaMemcpyAsync(num_triangles_out, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaDeviceSynchronize(); 

        dim3 blockSize(16, 16);
        dim3 gridSize(
            (iCols + blockSize.x - 1) / blockSize.x,
            (iRows + blockSize.y - 1) / blockSize.y
        );

        if (iRows <= 0 || iCols <= 0) {
            int zero_triangles = 0;
            cudaMemcpyAsync(num_triangles_out, &zero_triangles, sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaDeviceSynchronize();
            return;
        }

        createTrianglesFromNeighborsKernel<<<gridSize, blockSize, 0, stream>>>(
            Xw, Yw, Zw,
            iRows, iCols,
            four_neighbors,
            normals,
            (Triangle*)triangles_out, // 强制转换为 Triangle*
            num_triangles_out,
            dMaxEdgeLengthThreshold,
            dMaxNormalConststencyThreshold
        );
        cudaDeviceSynchronize();
    }
}
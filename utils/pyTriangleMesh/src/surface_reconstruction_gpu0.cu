#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <math.h>
#include <stdio.h>
#include <iostream>

#define MAX_TRIANGLES 20000
#define MAX_POINTS 8000
#define MAX_NEIGHBORS 50
#define EPSILON 1e-6

struct Point3D {
    float x, y, z;
    __device__ __host__ Point3D() : x(0), y(0), z(0) {}
    __device__ __host__ Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __device__ __host__ Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }

    __device__ __host__ Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.y);
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
        return n > EPSILON ? Point3D(x/n, y/n, z/n) : Point3D(0, 0, 0);
    }
};

struct Triangle {
    int v0, v1, v2;
    __device__ __host__ Triangle() : v0(-1), v1(-1), v2(-1) {}
    __device__ __host__ Triangle(int a, int b, int c) : v0(a), v1(b), v2(c) {}
};

// 计算两点之间的距离
__device__ float distance(const Point3D& p1, const Point3D& p2) {
    return sqrtf((p1.x - p2.x) * (p1.x - p2.x) +
                 (p1.y - p2.y) * (p1.y - p2.y) +
                 (p1.z - p2.z) * (p1.z - p2.z));
}

// K近邻搜索
__global__ void findKNearestNeighbors(Point3D* points, int num_points,
                                     int* neighbors, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    Point3D query_point = points[tid];
    float distances[MAX_NEIGHBORS];
    int indices[MAX_NEIGHBORS];

    for (int i = 0; i < k; i++) {
        distances[i] = FLT_MAX;
        indices[i] = -1;
    }

    for (int i = 0; i < num_points; i++) {
        if (i == tid) continue;

        float dist = distance(query_point, points[i]);

        for (int j = 0; j < k; j++) {
            if (dist < distances[j]) {
                for (int l = k - 1; l > j; l--) {
                    distances[l] = distances[l - 1];
                    indices[l] = indices[l - 1];
                }
                distances[j] = dist;
                indices[j] = i;
                break;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        neighbors[tid * k + i] = indices[i];
    }
}

// 计算点的法向量
__global__ void computeNormals(Point3D* points, int* neighbors,
                              Point3D* normals, int num_points, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    Point3D center_point = points[tid];

    Point3D centroid(0, 0, 0);
    int valid_neighbors = 0;

    for (int i = 0; i < k; i++) {
        int neighbor_idx = neighbors[tid * k + i];
        if (neighbor_idx >= 0 && neighbor_idx < num_points) {
            centroid = centroid + points[neighbor_idx];
            valid_neighbors++;
        }
    }

    if (valid_neighbors > 0) {
        centroid = centroid * (1.0f / valid_neighbors);
    }

    float cov[9] = {0};

    for (int i = 0; i < k; i++) {
        int neighbor_idx = neighbors[tid * k + i];
        if (neighbor_idx >= 0 && neighbor_idx < num_points) {
            Point3D diff = points[neighbor_idx] - centroid;

            cov[0] += diff.x * diff.x;
            cov[1] += diff.x * diff.y;
            cov[2] += diff.x * diff.z;
            cov[3] += diff.y * diff.x;
            cov[4] += diff.y * diff.y;
            cov[5] += diff.y * diff.z;
            cov[6] += diff.z * diff.x;
            cov[7] += diff.z * diff.y;
            cov[8] += diff.z * diff.z;
        }
    }

    Point3D v(1, 1, 1);
    float current_norm;

    for (int iter = 0; iter < 10; ++iter) {
        Point3D Av;
        Av.x = cov[0] * v.x + cov[1] * v.y + cov[2] * v.z;
        Av.y = cov[3] * v.x + cov[4] * v.y + cov[5] * v.z;
        Av.z = cov[6] * v.x + cov[7] * v.y + cov[8] * v.z;

        current_norm = Av.norm();
        if (current_norm > EPSILON) {
            v = Av * (1.0f / current_norm);
        } else {
            v = Point3D(0, 0, 0);
            break;
        }
    }
    normals[tid] = v;
}

// 基于法向量一致性的三角化
__global__ void triangulateWithNormals(Point3D* points, int* neighbors,
                                      Point3D* normals, Triangle* triangles,
                                      int* num_triangles, int num_points, int k, float alpha) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    Point3D center = points[tid];
    Point3D center_normal = normals[tid];

    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            int idx1 = neighbors[tid * k + i];
            int idx2 = neighbors[tid * k + j];

            if (idx1 < 0 || idx1 >= num_points ||
                idx2 < 0 || idx2 >= num_points) continue;

            Point3D p1 = points[idx1];
            Point3D p2 = points[idx2];

            Point3D edge1 = p1 - center;
            Point3D edge2 = p2 - center;
            Point3D tri_normal = edge1.cross(edge2);

            float tri_normal_norm = tri_normal.norm();
            if (tri_normal_norm < EPSILON) continue;
            tri_normal = tri_normal * (1.0f / tri_normal_norm);

            float normal_consistency = tri_normal.dot(center_normal);

            float edge1_len = edge1.norm();
            float edge2_len = edge2.norm();
            Point3D p1_p2_vec = p2 - p1;
            float edge3_len = p1_p2_vec.norm();

            float min_edge = fminf(fminf(edge1_len, edge2_len), edge3_len);
            float max_edge = fmaxf(fmaxf(edge1_len, edge2_len), edge3_len);
            float aspect_ratio = max_edge / (min_edge + EPSILON);
            printf("normal_consistency: %f, aspect_ratio: %f\n", normal_consistency, aspect_ratio);
            // std::abs(normal_consistency) > 0.2f && aspect_ratio < 5.0f && std::abs(normal_consistency) > 0.01f && 
            if (aspect_ratio < 5.0f && min_edge > EPSILON && max_edge < alpha) {
                int tri_idx = atomicAdd(num_triangles, 1);
                if (tri_idx < MAX_TRIANGLES) {
                    triangles[tri_idx] = Triangle(tid, idx1, idx2);
                }
            }
        }
    }
}

// Alpha Shape算法的简化版本
__global__ void alphaShapeTriangulation(Point3D* points, int* neighbors,
                                       Triangle* triangles, int* num_triangles,
                                       int num_points, int k, float alpha) {
    int tid = blockIdx.x * blockIdx.x + threadIdx.x;
    if (tid >= num_points) return;

    Point3D center = points[tid];

    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            int idx1 = neighbors[tid * k + i];
            int idx2 = neighbors[tid * k + j];

            if (idx1 < 0 || idx1 >= num_points ||
                idx2 < 0 || idx2 >= num_points) continue;

            Point3D p1 = points[idx1];
            Point3D p2 = points[idx2];

            Point3D a = center;
            Point3D b = p1;
            Point3D c = p2;

            float ab = (b - a).norm();
            float bc = (c - b).norm();
            float ca = (a - c).norm();

            // 计算三角形面积
            Point3D cross_vec = (b - a).cross(c - a);
            float tri_area = 0.5f * cross_vec.norm();

            // 计算三角形外接圆半径
            float circumradius = FLT_MAX; // 初始化为最大值
            if (std::abs(tri_area) > EPSILON) { // 避免除以零，并过滤掉退化三角形
                // 外接圆半径公式 R = (abc) / (4 * Area)
                circumradius = (ab * bc * ca) / (4.0f * tri_area);
            } else {
                // 如果面积为零或接近零，则三角形退化，不进行处理
                continue;
            }

            // Alpha Shape 核心条件：如果外接圆半径小于 alpha
            // 并且三角形面积有效 (再次检查，以防边缘情况)
            if (std::abs(circumradius) <= alpha) { // CloudCompare 的实现通常是 circumradius <= alpha
                                       // 根据实际需求可以调整为 <=
                int tri_idx = atomicAdd(num_triangles, 1);
                if (tri_idx < MAX_TRIANGLES) {
                    triangles[tri_idx] = Triangle(tid, idx1, idx2);
                }
            }
        }
    }
}

// 球枢轴算法 (Ball Pivoting Algorithm) 的简化版本
// 球枢轴算法 (Ball Pivoting Algorithm) 的简化版本
__global__ void ballPivotingTriangulation(Point3D* points, int* neighbors,
                                         Point3D* normals, Triangle* triangles,
                                         int* num_triangles, int num_points,
                                         int k, float ball_radius) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    Point3D center = points[tid];
    Point3D center_normal = normals[tid];

    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            int idx1 = neighbors[tid * k + i];
            int idx2 = neighbors[tid * k + j];

            if (idx1 < 0 || idx1 >= num_points ||
                idx2 < 0 || idx2 >= num_points) continue;

            Point3D p1 = points[idx1];
            Point3D p2 = points[idx2];

            // 计算三角形边长
            float edge_len1 = (p1 - center).norm();
            float edge_len2 = (p2 - center).norm();
            Point3D p1_p2_vec = p2 - p1; // <--- 定义 p1_p2_vec
            float edge3_len = p1_p2_vec.norm(); // <--- 定义并计算 edge3_len

            float max_edge = fmaxf(fmaxf(edge_len1, edge_len2), edge3_len); // 现在 edge3_len 已定义

            Point3D tri_normal = (p1 - center).cross(p2 - center);
            float tri_normal_norm = tri_normal.norm();
            if (tri_normal_norm < EPSILON) continue;
            tri_normal = tri_normal * (1.0f / tri_normal_norm);

            if (max_edge < 2.0f * ball_radius &&
                tri_normal.dot(center_normal) > 0.1f) {

                int tri_idx = atomicAdd(num_triangles, 1);
                if (tri_idx < MAX_TRIANGLES) {
                    triangles[tri_idx] = Triangle(tid, idx1, idx2);
                }
            }
        }
    }
}


// 新的核函数，用于将 float* 转换为 Point3D* (在设备上)
__global__ void convertFloatToPoint3D(float* src_float_ptr, Point3D* dst_point3d_ptr, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        dst_point3d_ptr[tid] = Point3D(src_float_ptr[tid * 3], src_float_ptr[tid * 3 + 1], src_float_ptr[tid * 3 + 2]);
    }
}

// 新的核函数，用于将 Triangle* 数组（GPU）转换为 int* 数组（GPU）
__global__ void convertTrianglesToInt(Triangle* src_triangles, int* dst_int_triangles, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        dst_int_triangles[tid * 3] = src_triangles[tid].v0;
        dst_int_triangles[tid * 3 + 1] = src_triangles[tid].v1;
        dst_int_triangles[tid * 3 + 2] = src_triangles[tid].v2;
    }
}


extern "C" {
    void surface_reconstruction_cuda(float* points, int num_points,
                                   int* triangles, int* num_triangles, // triangles 和 num_triangles 是 GPU 输出指针
                                   int algorithm_type, float param,
                                   cudaStream_t stream) {
        Point3D* d_points;
        Point3D* d_normals;
        Triangle* d_triangles;      // 临时 GPU 存储 Triangle 结构体
        int* d_neighbors;          // <--- 添加这行声明
        int* d_num_triangles_temp;  // 临时 GPU 存储计数

        int k = fminf(3.0f, (float)num_points - 1.0f);
        if (k < 1) k = 1;

        // 分配临时 GPU 内存
        std::cout << "分配临时 GPU 内存" << std::endl;
        cudaMalloc(&d_points, num_points * sizeof(Point3D));
        cudaMalloc(&d_normals, num_points * sizeof(Point3D));
        cudaMalloc(&d_triangles, MAX_TRIANGLES * sizeof(Triangle));
        cudaMalloc(&d_neighbors, num_points * k * sizeof(int)); // 现在 d_neighbors 已定义
        cudaMalloc(&d_num_triangles_temp, sizeof(int)); // 用于临时存储三角形数量

        dim3 blockSize(256);
        dim3 gridSize((num_points + blockSize.x - 1) / blockSize.x);
        
        std::cout << "调用新的 __global__ 函数进行 float* 到 Point3D* 的转换" << std::endl;

        // 调用新的 __global__ 函数进行 float* 到 Point3D* 的转换
        convertFloatToPoint3D<<<gridSize, blockSize, 0, stream>>>(points, d_points, num_points);
        cudaDeviceSynchronize();

        std::cout << "初始化三角形计数" << std::endl;
        int zero = 0;
        cudaMemcpyAsync(d_num_triangles_temp, &zero, sizeof(int), cudaMemcpyHostToDevice, stream); // 初始化临时计数
        cudaDeviceSynchronize(); // 确保初始化完成

        std::cout << "第一步：找K近邻" << std::endl;
        findKNearestNeighbors<<<gridSize, blockSize, 0, stream>>>(d_points, num_points, d_neighbors, k);
        cudaDeviceSynchronize();

        std::cout << "第二步：计算法向量" << std::endl;
        computeNormals<<<gridSize, blockSize, 0, stream>>>(d_points, d_neighbors, d_normals, num_points, k);
        cudaDeviceSynchronize();

        std::cout << "第三步：根据算法类型进行三角化" << std::endl;
        switch (algorithm_type) {
            case 0:
                std::cout << "基于法向量的三角化" << std::endl;
                triangulateWithNormals<<<gridSize, blockSize, 0, stream>>>(
                    d_points, d_neighbors, d_normals, d_triangles,
                    d_num_triangles_temp, num_points, k, param); // 写入临时计数
                break;
            case 1:
                std::cout << "Alpha Shape" << std::endl;
                alphaShapeTriangulation<<<gridSize, blockSize, 0, stream>>>(
                    d_points, d_neighbors, d_triangles, d_num_triangles_temp, // 写入临时计数
                    num_points, k, param);
                break;
            case 2:
                std::cout << "Ball Pivoting" << std::endl;
                ballPivotingTriangulation<<<gridSize, blockSize, 0, stream>>>(
                    d_points, d_neighbors, d_normals, d_triangles,
                    d_num_triangles_temp, num_points, k, param); // 写入临时计数
                break;
        }

        cudaDeviceSynchronize(); // 确保所有核函数执行完毕

        // === 关键修改：将临时 GPU 结果复制到最终的 GPU 输出指针 ===
        // 1. 将最终的三角形数量从临时 GPU 计数复制到输出 GPU 指针
        cudaMemcpyAsync(num_triangles, d_num_triangles_temp, sizeof(int), cudaMemcpyDeviceToDevice, stream);
        
        // 2. 从临时 GPU 存储中获取实际三角形数量，以便启动转换核函数
        //    这里需要将 d_num_triangles_temp 复制回 CPU 以获取实际的 count
        //    或者，如果 num_triangles 已经是 final count，可以直接使用其值。
        //    为了避免 H2D 拷贝，我们假设此时 num_triangles 已经包含了正确的值 (从上一步复制过来)
        //    但更安全地做法是：先 H2D 拷贝到 h_final_num_triangles，再用其启动 kernel
        
        int h_final_num_triangles;
        cudaMemcpyAsync(&h_final_num_triangles, d_num_triangles_temp, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaDeviceSynchronize(); // 确保计数已经复制回 CPU

        // 3. 启动核函数将 Triangle* 数据转换为 int* 格式并写入输出 GPU 指针
        if (h_final_num_triangles > 0) {
            dim3 convert_output_grid((h_final_num_triangles + blockSize.x - 1) / blockSize.x);
            convertTrianglesToInt<<<convert_output_grid, blockSize, 0, stream>>>(
                d_triangles, triangles, h_final_num_triangles);
            cudaDeviceSynchronize(); // 确保转换完成
        }

        std::cout << "清理内存" << std::endl;
        // 清理所有分配的 GPU 内存
        cudaFree(d_points);
        cudaFree(d_normals);
        cudaFree(d_triangles);
        cudaFree(d_neighbors);
        cudaFree(d_num_triangles_temp); // 释放临时计数内存
    }
}
#include <torch/extension.h>
#include <vector>
#include <iostream> // 用于调试输出
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/detail/KernelUtils.h>  // if needed
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <torch/extension.h>
#include <vector>
#include <iostream>

// Define MAX_TRIANGLES here as well, to ensure consistency with the .cu file
// and make it visible to the C++ wrapper.
#define MAX_TRIANGLES 2000000 

// 声明在 surface_reconstruction_gpu.cu 中定义的主机端 C 函数
extern "C" void find_four_grid_neighbors_cuda_wrapper(
    const float* Xw, const float* Yw, const float* Zw,
    int iRows, int iCols,
    int* output_neighbors,
    cudaStream_t stream);

// 新增的法向量计算函数声明
extern "C" void calculate_normals_cuda_wrapper(
    const float* Xw, const float* Yw, const float* Zw,
    int iRows, int iCols,
    const int* four_neighbors,
    float* normals_out,
    cudaStream_t stream);

extern "C" void create_triangles_from_neighbors_cuda_wrapper(
    const float* Xw, const float* Yw, const float* Zw,
    int iRows, int iCols,
    const int* four_neighbors,
    const float* normals,
    int* triangles_out,
    int* num_triangles_out,
    float dMaxEdgeLengthThreshold,
    float dMaxNormalConststencyThreshold,
    cudaStream_t stream);

// 辅助宏用于检查 CUDA Tensor 和连续性
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor!")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous!")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Stage 1 Wrapper: 寻找四个方向的邻居
torch::Tensor find_four_grid_neighbors(
    torch::Tensor Xw_tensor,          // (rows, cols) 浮点型 Tensor
    torch::Tensor Yw_tensor,          // (rows, cols) 浮点型 Tensor
    torch::Tensor Zw_tensor           // (rows, cols) 浮点型 Tensor
) {
    CHECK_INPUT(Xw_tensor);
    CHECK_INPUT(Yw_tensor);
    CHECK_INPUT(Zw_tensor);

    TORCH_CHECK(Xw_tensor.dtype() == torch::kFloat32, "Xw_tensor must be float32");
    TORCH_CHECK(Yw_tensor.dtype() == torch::kFloat32, "Yw_tensor must be float32");
    TORCH_CHECK(Zw_tensor.dtype() == torch::kFloat32, "Zw_tensor must be float32");

    TORCH_CHECK(Xw_tensor.dim() == 2, "Xw_tensor must be a 2D tensor.");
    TORCH_CHECK(Yw_tensor.dim() == 2, "Yw_tensor must be a 2D tensor.");
    TORCH_CHECK(Zw_tensor.dim() == 2, "Zw_tensor must be a 2D tensor.");

    TORCH_CHECK(Xw_tensor.sizes() == Yw_tensor.sizes() && Xw_tensor.sizes() == Zw_tensor.sizes(), 
                "Xw_tensor, Yw_tensor, and Zw_tensor must have the same shape.");

    int iRows = Xw_tensor.size(0);
    int iCols = Xw_tensor.size(1);
    int total_points = iRows * iCols;

    const float* Xw_ptr_device = Xw_tensor.data<float>();
    const float* Yw_ptr_device = Yw_tensor.data<float>();
    const float* Zw_ptr_device = Zw_tensor.data<float>();

    // 输出 Tensor，每个点存储 4 个邻居索引 (Up, Down, Left, Right)
    torch::Tensor output_neighbors_tensor = torch::zeros(
        {iRows, iCols, 4}, // (rows, cols, 4)
        torch::TensorOptions().dtype(torch::kInt32).device(Xw_tensor.device())
    );
    int* output_neighbors_ptr_device = output_neighbors_tensor.data<int>();

    auto stream = at::cuda::getCurrentCUDAStream();

    find_four_grid_neighbors_cuda_wrapper(
        Xw_ptr_device, Yw_ptr_device, Zw_ptr_device,
        iRows, iCols,
        output_neighbors_ptr_device,
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after find_four_grid_neighbors_cuda_wrapper: " << cudaGetErrorString(err) << std::endl;
        TORCH_CHECK(false, "CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after device sync (find_four_grid_neighbors): " << cudaGetErrorString(err) << std::endl;
        TORCH_CHECK(false, "CUDA Sync Error: " + std::string(cudaGetErrorString(err)));
    }

    return output_neighbors_tensor;
}

// Stage 2 Wrapper: 计算法向量
torch::Tensor calculate_normals_from_grid(
    torch::Tensor Xw_tensor,          // (rows, cols) 浮点型 Tensor
    torch::Tensor Yw_tensor,          // (rows, cols) 浮点型 Tensor
    torch::Tensor Zw_tensor,          // (rows, cols) 浮点型 Tensor
    torch::Tensor four_neighbors_tensor // (rows, cols, 4) 整型 Tensor, 来自 find_four_grid_neighbors
) {
    CHECK_INPUT(Xw_tensor);
    CHECK_INPUT(Yw_tensor);
    CHECK_INPUT(Zw_tensor);
    CHECK_INPUT(four_neighbors_tensor);

    TORCH_CHECK(Xw_tensor.dtype() == torch::kFloat32, "Xw_tensor must be float32");
    TORCH_CHECK(Yw_tensor.dtype() == torch::kFloat32, "Yw_tensor must be float32");
    TORCH_CHECK(Zw_tensor.dtype() == torch::kFloat32, "Zw_tensor must be float32");
    TORCH_CHECK(four_neighbors_tensor.dtype() == torch::kInt32, "four_neighbors_tensor must be int32");

    TORCH_CHECK(Xw_tensor.dim() == 2, "Xw_tensor must be a 2D tensor.");
    TORCH_CHECK(Yw_tensor.dim() == 2, "Yw_tensor must be a 2D tensor.");
    TORCH_CHECK(Zw_tensor.dim() == 2, "Zw_tensor must be a 2D tensor.");
    TORCH_CHECK(four_neighbors_tensor.dim() == 3 && four_neighbors_tensor.size(2) == 4, 
                "four_neighbors_tensor must be (rows, cols, 4).");

    TORCH_CHECK(Xw_tensor.sizes() == Yw_tensor.sizes() && Xw_tensor.sizes() == Zw_tensor.sizes() &&
                Xw_tensor.size(0) == four_neighbors_tensor.size(0) && Xw_tensor.size(1) == four_neighbors_tensor.size(1),
                "All input tensors must have consistent grid dimensions.");

    int iRows = Xw_tensor.size(0);
    int iCols = Xw_tensor.size(1);

    const float* Xw_ptr_device = Xw_tensor.data<float>();
    const float* Yw_ptr_device = Yw_tensor.data<float>();
    const float* Zw_ptr_device = Zw_tensor.data<float>();
    const int* four_neighbors_ptr_device = four_neighbors_tensor.data<int>();

    // 输出法向量 Tensor，每个点存储 3 个分量 (x, y, z)
    torch::Tensor normals_tensor = torch::zeros(
        {iRows, iCols, 3}, // (rows, cols, 3)
        torch::TensorOptions().dtype(torch::kFloat32).device(Xw_tensor.device())
    );
    float* normals_ptr_device = normals_tensor.data<float>();

    auto stream = at::cuda::getCurrentCUDAStream();

    calculate_normals_cuda_wrapper(
        Xw_ptr_device, Yw_ptr_device, Zw_ptr_device,
        iRows, iCols,
        four_neighbors_ptr_device,
        normals_ptr_device,
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after calculate_normals_cuda_wrapper: " << cudaGetErrorString(err) << std::endl;
        TORCH_CHECK(false, "CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after device sync (calculate_normals_from_grid): " << cudaGetErrorString(err) << std::endl;
        TORCH_CHECK(false, "CUDA Sync Error: " + std::string(cudaGetErrorString(err)));
    }

    return normals_tensor;
}


// Stage 3 Wrapper: 根据邻居信息创建三角形
torch::Tensor create_triangles_from_grid_neighbors(
    torch::Tensor Xw_tensor,          // (rows, cols) 浮点型 Tensor
    torch::Tensor Yw_tensor,          // (rows, cols) 浮点型 Tensor
    torch::Tensor Zw_tensor,          // (rows, cols) 浮点型 Tensor
    torch::Tensor four_neighbors_tensor, // (rows, cols, 4) 整型 Tensor, 来自 find_four_grid_neighbors
    torch::Tensor normals_tensor, // (rows, cols, 3) 浮点型 Tensor, 来自 find_four_grid_neighbors
    float dMaxEdgeLengthThreshold,      // 最大边长阈值
    float dMaxNormalConststencyThreshold      // 最大边长阈值
) {
    CHECK_INPUT(Xw_tensor);
    CHECK_INPUT(Yw_tensor);
    CHECK_INPUT(Zw_tensor);
    CHECK_INPUT(normals_tensor);
    CHECK_INPUT(four_neighbors_tensor);

    TORCH_CHECK(Xw_tensor.dtype() == torch::kFloat32, "Xw_tensor must be float32");
    TORCH_CHECK(Yw_tensor.dtype() == torch::kFloat32, "Yw_tensor must be float32");
    TORCH_CHECK(Zw_tensor.dtype() == torch::kFloat32, "Zw_tensor must be float32");
    TORCH_CHECK(normals_tensor.dtype() == torch::kFloat32, "normals_tensor must be float32");
    TORCH_CHECK(four_neighbors_tensor.dtype() == torch::kInt32, "four_neighbors_tensor must be int32");

    TORCH_CHECK(Xw_tensor.dim() == 2, "Xw_tensor must be a 2D tensor.");
    TORCH_CHECK(Yw_tensor.dim() == 2, "Yw_tensor must be a 2D tensor.");
    TORCH_CHECK(Zw_tensor.dim() == 2, "Zw_tensor must be a 2D tensor.");
    TORCH_CHECK(normals_tensor.dim() == 3 && normals_tensor.size(2) == 3, 
                "normals_tensor must be (rows, cols, 3).");
    TORCH_CHECK(four_neighbors_tensor.dim() == 3 && four_neighbors_tensor.size(2) == 4, 
                "four_neighbors_tensor must be (rows, cols, 4).");

    TORCH_CHECK(Xw_tensor.sizes() == Yw_tensor.sizes() && Xw_tensor.sizes() == Zw_tensor.sizes() &&
                Xw_tensor.size(0) == four_neighbors_tensor.size(0)&& Xw_tensor.size(1) == normals_tensor.size(1) && Xw_tensor.size(1) == four_neighbors_tensor.size(1),
                "All input tensors must have consistent grid dimensions.");

    int iRows = Xw_tensor.size(0);
    int iCols = Xw_tensor.size(1);

    const float* Xw_ptr_device = Xw_tensor.data<float>();
    const float* Yw_ptr_device = Yw_tensor.data<float>();
    const float* Zw_ptr_device = Zw_tensor.data<float>();
    const float* normals_ptr_device = normals_tensor.data<float>();
    const int* four_neighbors_ptr_device = four_neighbors_tensor.data<int>();

    // 预估最大三角形数量，与 .cu 文件中的 MAX_TRIANGLES 保持一致
    const int ESTIMATED_MAX_TRIANGLES = MAX_TRIANGLES; 

    torch::Tensor triangles_tensor = torch::zeros(
        {ESTIMATED_MAX_TRIANGLES, 3},
        torch::TensorOptions().dtype(torch::kInt32).device(Xw_tensor.device())
    );
    int* triangles_ptr_device = triangles_tensor.data<int>();

    torch::Tensor num_triangles_tensor = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kInt32).device(Xw_tensor.device())
    );
    int* num_triangles_ptr_device = num_triangles_tensor.data<int>();

    auto stream = at::cuda::getCurrentCUDAStream();

    create_triangles_from_neighbors_cuda_wrapper(
        Xw_ptr_device, Yw_ptr_device, Zw_ptr_device,
        iRows, iCols,
        four_neighbors_ptr_device,
        normals_ptr_device,
        triangles_ptr_device,
        num_triangles_ptr_device,
        dMaxEdgeLengthThreshold,
        dMaxNormalConststencyThreshold,
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after create_triangles_from_neighbors_cuda_wrapper: " << cudaGetErrorString(err) << std::endl;
        TORCH_CHECK(false, "CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after device sync (create_triangles_from_grid_neighbors): " << cudaGetErrorString(err) << std::endl;
        TORCH_CHECK(false, "CUDA Sync Error: " + std::string(cudaGetErrorString(err)));
    }

    int final_num_triangles = num_triangles_tensor.item<int>();
    return triangles_tensor.slice(0, 0, final_num_triangles);
}

// PYBIND11 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("find_four_grid_neighbors", &find_four_grid_neighbors,
          "Find nearest up, down, left, right neighbors for grid points (CUDA)");
    m.def("calculate_normals_from_grid", &calculate_normals_from_grid,
          "Calculate surface normals for grid points using four nearest neighbors (CUDA)");
    m.def("surface_reconstruction", &create_triangles_from_grid_neighbors,
          "Create mesh triangles from grid points and their four nearest neighbors (CUDA)");
}

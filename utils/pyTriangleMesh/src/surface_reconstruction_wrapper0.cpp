#include <torch/extension.h>
#include <vector>
#include <iostream> // 用于调试输出
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/detail/KernelUtils.h>  // if needed
#include <cuda.h>
#include <cuda_runtime_api.h>


// 假设 Point3D 结构体在某个头文件中定义，或者已经通过其他方式可用
// 例如，如果它在 surface_reconstruction_gpu.cu 或其包含的头文件中，
// 并且你需要在这里使用它，确保它被包含或声明。
// 如果它仅用于 CUDA 核函数内部，则无需在此处可见。

// 声明在 surface_reconstruction_gpu.cu 中定义的 C 函数
// === 核心修改：更新函数声明以匹配 surface_reconstruction_cuda 的新签名 ===
extern "C" void surface_reconstruction_cuda(float* points_ptr_device, int num_points,
                                            int* triangles_ptr_device, int* num_triangles_ptr_device,
                                            int algorithm_type, float param,
                                            cudaStream_t stream); // <--- 添加 stream 参数

// 定义一个宏用于简化输入张量检查，类似于 CHECK_INPUT
// PyTorch 官方扩展通常使用 TORCH_CHECK 来实现这些检查
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor!")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous!")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor surface_reconstruction_wrapper(
    torch::Tensor points_tensor,          // (N, 3) 浮点型 Tensor
    int algorithm_type,            // 算法类型
    float param                    // 算法参数
) {
    // 使用 CHECK_INPUT 宏进行输入张量检查
    CHECK_INPUT(points_tensor);

    // 针对 surface_reconstruction_wrapper 的特定张量形状和DType检查
    TORCH_CHECK(points_tensor.dtype() == torch::kFloat32, "points_tensor must be float32");
    TORCH_CHECK(points_tensor.dim() == 2 && points_tensor.size(1) == 3, "points_tensor must be a (N, 3) tensor");

    int num_points = points_tensor.size(0);

    // 直接通过 .data<T>() 获取设备（GPU）数据指针
    float* points_ptr_device = points_tensor.data<float>();

    // 预估最大三角形数量。根据 surface_reconstruction_gpu.cu 中的 MAX_TRIANGLES 定义
    const int MAX_TRIANGLES_OUTPUT = 20000; // 示例值，请与 .cu 文件中的实际定义保持一致

    // 创建输出三角形张量 (在 GPU 上)
    torch::Tensor triangles_tensor = torch::zeros(
        {MAX_TRIANGLES_OUTPUT, 3},
        torch::TensorOptions().dtype(torch::kInt32).device(points_tensor.device())
    );
    int* triangles_ptr_device = triangles_tensor.data<int>();

    // 创建一个标量 Tensor 用于存储最终的三角形数量 (在 GPU 上)
    torch::Tensor num_triangles_tensor = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kInt32).device(points_tensor.device())
    );
    int* num_triangles_ptr_device = num_triangles_tensor.data<int>();

    // 获取当前 CUDA 流
    auto stream = at::cuda::getCurrentCUDAStream();

    // 调用 CUDA 函数
    std::cout << "调用 surface_reconstruction_cuda 函数" << std::endl;
    surface_reconstruction_cuda(
        points_ptr_device,        // GPU 指针
        num_points,
        triangles_ptr_device,     // GPU 指针
        num_triangles_ptr_device, // GPU 指针
        algorithm_type,
        param,
        stream                    // <--- 传递 stream 参数
    );

    // 强制同步，确保所有 CUDA 操作完成并捕获潜在的异步错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after surface_reconstruction_cuda: " << cudaGetErrorString(err) << std::endl;
        TORCH_CHECK(false, "CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after device sync: " << cudaGetErrorString(err) << std::endl;
        TORCH_CHECK(false, "CUDA Sync Error: " + std::string(cudaGetErrorString(err)));
    }

    // 获取实际的三角形数量 (从 GPU 复制回 CPU)
    int final_num_triangles = num_triangles_tensor.item<int>();
    std::cout<< final_num_triangles << std::endl;
    // 返回实际包含数据的部分
    return triangles_tensor.slice(0, 0, final_num_triangles);
}

// PYBIND11 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("surface_reconstruction", &surface_reconstruction_wrapper,
          "Surface reconstruction (CUDA)");
}
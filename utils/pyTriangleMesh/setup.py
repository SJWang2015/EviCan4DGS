# setup.py
import os
import sys
import torch
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

base_dir = os.path.abspath(os.path.dirname(__file__))

class VerboseBuildExt(BuildExtension):
    def build_extensions(self):
        print("开始构建CUDA扩展...")
        
        # 打印环境信息
        print(f"Python: {sys.executable}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device count: {torch.cuda.device_count()}")
        
        # 检查nvcc
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            print(f"nvcc version: {result.stdout.split('release')[1].split(',')[0].strip()}")
        except:
            print("nvcc not found!")
        
        # 检查环境变量
        cuda_home = os.environ.get('CUDA_HOME', 'Not set')
        print(f"CUDA_HOME: {cuda_home}")
        
        try:
            super().build_extensions()
            print("CUDA扩展构建成功!")
        except Exception as e:
            print(f"构建失败: {e}")
            # 这里可以添加CPU备用构建逻辑
            raise

def get_cuda_extension():
    """获取CUDA扩展，添加更多调试信息"""
    print("配置CUDA扩展...")
    
    # 更改源文件为 surface_reconstruction_gpu.cu 和 surface_reconstruction_wrapper.cpp
    # 假设这些文件位于 setup.py 同级的 'src' 目录下
    sources = [
        os.path.join(base_dir, 'src', 'surface_reconstruction_gpu.cu'),
        os.path.join(base_dir, 'src', 'surface_reconstruction_wrapper.cpp'),
    ]
    
    # 检查源文件是否存在
    for src_file in sources:
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Source file not found: {src_file}. Please ensure it's in the 'src' directory.")
        print(f"找到源文件: {src_file}")
    
    # 编译参数
    extra_compile_args = {
        # 'cxx': ['-O3', '-std=c++14'],
        # 'nvcc': ['-O3', '--use_fast_math', '--expt-relaxed-constexpr']
        'cxx': ['-g'],
        'nvcc': ['-O2']
    }
    
    # 根据CUDA版本调整参数
    try:
        cuda_version = torch.version.cuda
        if cuda_version:
            major_version = int(cuda_version.split('.')[0])
            if major_version >= 11:
                # 针对较新CUDA版本添加多架构支持
                extra_compile_args['nvcc'].extend([
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86'
                ])
            print(f"为CUDA {cuda_version}配置编译参数")
    except Exception as e:
        print(f"无法确定CUDA版本或配置参数失败: {e}，使用默认参数")
    
    return [
        CUDAExtension(
            name='pyTriangleMesh.triMesh', # 模块名称保持不变，你将导入 pyTriangleMesh._C
            sources=sources,
            extra_compile_args=extra_compile_args,
            libraries=['cuda', 'cudart'], # 确保这些库被链接
        )
    ]

# 主setup
setup(
    name="pyTriangleMesh",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=get_cuda_extension() if torch.cuda.is_available() else [],
    cmdclass={'build_ext': VerboseBuildExt} if torch.cuda.is_available() else {},
    install_requires=[
        'torch>=1.8.0',
        'numpy>=1.19.0',
    ],
    zip_safe=False,
)
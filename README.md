<p align="center">
    <!-- project -->
    <a href="https://github.com/SJWang2015/EviCan4DGS/"></a>
</p>

<p align="center">
  <img src="https://github.com/SJWang2015/EviCan4DGS/media/output.gif" width="49%" style="max-width: 100%; height: auto;" />
  <img src="https://github.com/SJWang2015/EviCan4DGS/media/output2.gif" width="49%" style="max-width: 100%; height: auto;" />
</p>

## About
# EviCan4DGS

EviCan4DGS is a 4D Gaussian Splatting framework for dynamic scene reconstruction and novel view synthesis in autonomous-driving environments. Developed based on [DriveStudio](https://ziyc.github.io/omnire/), it is designed for complex traffic scenes with sparse multi-view observations, and focuses on improving the modeling of both non-rigid humans and rigid vehicles.
Our method combines SMPL-driven human deformation modeling with structural-evidence-guided adaptive refinement for vehicles, enabling better primitive allocation and more accurate geometry reconstruction under non-uniform point-cloud density. EviCan4DGS also introduces a priority-based densification and pruning strategy for stable optimization within a fixed primitive budget.
Experiments on the [Waymo Open Dataset](https://waymo.com/open/) show that EviCan4DGS outperforms existing methods in reconstruction quality and generalization across diverse driving scenes.

## 🔨 Installation

Run the following commands to set up the environment:

```shell
# Clone the repository with submodules
git clone --recursive https://github.com/ziyc/drivestudio.git
cd drivestudio

# Create the environment
conda create -n drivestudio python=3.9 -y
conda activate drivestudio
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast

CFLAGS="-I/root/anaconda3/envs/citygs/include/python3.9" \
CXXFLAGS="-I/root/anaconda3/envs/citygs/include/python3.9" \
TORCH_CUDA_ARCH_LIST="8.0;8.9" \
pip install git+https://github.com/NVlabs/nvdiffrast
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast

cuobjdump --list-elf /root/anaconda3/envs/citygs/lib/python3.9/site-packages/pytorch3d/_C.cpython-39-x86_64-linux-gnu.so | grep sm_

python setup.py build_ext --inplace -v
pip install git+https://github.com/NVlabs/nvdiffrast

pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install .
pip install git+https://github.com/NVlabs/nvdiffrast
pip install .
pip install git+https://github.com/facebookresearch/pytorch3d.git

# Set up for SMPL Gaussians
cd third_party/smplx/
pip install -e .
cd ../..


CFLAGS="-I/root/anaconda3/envs/citygs/include/python3.9" \
CXXFLAGS="-I/root/anaconda3/envs/citygs/include/python3.9" \
TORCH_CUDA_ARCH_LIST="8.0;8.9" \
pip install -v --no-cache-dir .

export TORCH_CUDA_ARCH_LIST="8.0;8.9"
pip uninstall -y pointnet2 2>/dev/null || true
python setup.py clean
rm -rf build
pip install -v --no-cache-dir .


# Install pybcpd
cd third_party/pybcpd/
# Make sure you have install liblapack-dev & libblas-dev before installation. 
# If not, you can run "sudo apt install liblapack-dev libblas-dev"
# python setup.py build_ext --inplace
# Example your virtual python environment is "/root/anaconda3/envs/citygs" and python version is "python3.9"
CFLAGS="-I/root/anaconda3/envs/citygs/include/python3.9" \
CXXFLAGS="-I/root/anaconda3/envs/citygs/include/python3.9" \
TORCH_CUDA_ARCH_LIST="8.0;8.9" \
python setup.py build_ext --inplace -v

pip install . 

```

## 🚀 Running
### Training
```shell
export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

python tools/train.py \
    --config_file configs/evican4dgs.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/3cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```



- To run other methods, change `--config_file`. See `configs/` for more options.
- Specify dataset and number of cameras by setting `dataset`. Examples: `waymo/1cams`, `waymo/5cams`, `pandaset/6cams`, `argoverse/7cams`, etc.
  You can set up arbitrary camera combinations for each dataset. See `configs/datasets/` for custom configuration details.
- For over 3 cameras or 450+ images, we recommend using `omnire_extended_cam.yaml`. It works better in practice.
### Evaluation
```shell
python tools/eval.py --resume_from $ckpt_path
```

## 🙏 Acknowledgments
We utilize the rasterization kernel from [gsplat](https://github.com/nerfstudio-project/gsplat). Parts of our implementation are based on work from [OmniRe](https://ziyc.github.io/omnire/), [EmerNeRF](https://github.com/NVlabs/EmerNeRF), [NerfStudio](https://github.com/nerfstudio-project/nerfstudio), [GART](https://github.com/JiahuiLei/GART), and [Neuralsim](https://github.com/PJLab-ADG/neuralsim). Implementations related to [Deformable-GS](https://github.com/ingra14m/Deformable-3D-Gaussians), [PVG](https://github.com/fudan-zvg/PVG), and [Street Gaussians](https://github.com/zju3dv/street_gaussians) are inherited from the [OmniRe](https://ziyc.github.io/omnire/) / DriveStudio framework, with reference to their original codebases.

We extend our sincere gratitude to the authors of these projects for their valuable contributions to the community, which have greatly supported our research.


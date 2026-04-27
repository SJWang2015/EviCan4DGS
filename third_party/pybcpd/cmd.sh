# Uninstall 
pip uninstall pybcpd -y

# Clear buidl files
rm -rf build/ dist/ *.egg-info/
rm -rf *.so *.pyd

# Install 
# Make sure you have install liblapack-dev & libblas-dev before installation. 
# If not, you can run "sudo apt install liblapack-dev libblas-dev"
# python setup.py build_ext --inplace
# Example your virtual python environment is "/root/anaconda3/envs/citygs" and python version is "python3.9"
CFLAGS="-I/root/anaconda3/envs/citygs/include/python3.9" \
CXXFLAGS="-I/root/anaconda3/envs/citygs/include/python3.9" \
TORCH_CUDA_ARCH_LIST="8.0;8.9" \
python setup.py build_ext --inplace -v

pip install . 

# This Script Assumes Python 3.8, CUDA 11.3, PyTorch 1.11.0

conda deactivate

# Set environment variables
export ENV_NAME=SGLoc
export PYTHON_VERSION=3.8
export PYTORCH_VERSION=1.11.0
export CUDA_VERSION=11.3

# Create a new conda environment and activate it
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
conda activate $ENV_NAME

# Install PyTorch and a compatible NumPy version
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 "numpy<1.23" -c pytorch -y

# Install TorchSparse (this is much simpler than MinkowskiEngine)
pip install torchsparse

# Install the rest of the packages
pip install matplotlib
pip install h5py
pip install opencv-python
pip install pandas
pip install transforms3d
pip install open3d
pip install tensorboardX

echo "✅ Installation complete. Environment '$ENV_NAME' with TorchSparse is ready."

# # This Script Assumes Python 3.9, CUDA 11.6

# conda deactivate

# # Set environment variables
# export ENV_NAME=SGLoc
# export PYTHON_VERSION=3.8
# export PYTORCH_VERSION=1.11.0
# export CUDA_VERSION=11.3

# # Create a new conda environment and activate it
# conda create -n $ENV_NAME python=$PYTHON_VERSION -y
# conda activate $ENV_NAME

# # Install PyTorch
# # conda install pytorch==$PYTORCH_VERSION torchvision torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=$CUDA_VERSION -c pytorch -y
# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 "numpy<1.23" -c pytorch -y
# # Install MinkowskiEngine
# conda install openblas-devel -c anaconda -y
# conda install -y -c nvidia/label/cuda-11.3.0 cuda-nvcc

# export CUDA_HOME="$CONDA_PREFIX"
# export PATH="$CONDA_PREFIX/bin:$PATH"

# pip install pip==22.2.1
# # pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
# # Install
# pip install matplotlib
# pip install h5py
# pip install opencv-python
# pip install pandas
# pip install transforms3d
# pip install open3d
# pip install tensorboardX

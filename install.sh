# This Script Assumes Python 3.8, CUDA 11.3 for PyTorch 1.11.0

conda deactivate

# Set environment variables
export ENV_NAME=SGLoc
export PYTHON_VERSION=3.8
export PYTORCH_VERSION=1.11.0
export CUDA_VERSION=11.3

# Create a new conda environment and activate it
echo "--- Creating new conda environment: $ENV_NAME ---"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
conda activate $ENV_NAME

# Install PyTorch from the official pytorch channel
echo "--- Installing PyTorch $PYTORCH_VERSION with CUDA $CUDA_VERSION ---"
conda install pytorch==$PYTORCH_VERSION torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=$CUDA_VERSION -c pytorch -y

# Install build dependencies required by MinkowskiEngine
echo "--- Installing build dependencies (openblas, g++, ninja, cuda-toolkit) ---"
conda install openblas-devel -c anaconda -y
conda install gxx_linux-64 -y
conda install ninja -y                                  # Recommended build system for PyTorch extensions
conda install -c nvidia cuda-toolkit=11.3 -y            # CRITICAL FIX: Installs the nvcc compiler

# Install compatible build tools (setuptools<60)
echo "--- Installing compatible pip build tools ---"
pip install --upgrade pip "setuptools<60" wheel

# Install MinkowskiEngine from PyPI
echo "--- Installing MinkowskiEngine ---"
pip install MinkowskiEngine --no-deps

# Install the rest of the packages
echo "--- Installing remaining packages ---"
pip install matplotlib h5py opencv-python pandas transforms3d open3d tensorboardX

echo ""
echo "✅ Installation complete. Environment '$ENV_NAME' is ready and active."

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
# conda install pytorch==$PYTORCH_VERSION torchvision torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=$CUDA_VERSION -c pytorch -y
# # Install MinkowskiEngine
# conda install openblas-devel -c anaconda -y
# conda install -y -c nvidia/label/cuda-11.3.0 cuda-nvcc

# export CUDA_HOME="$CONDA_PREFIX"
# export PATH="$CONDA_PREFIX/bin:$PATH"

# pip install pip==22.2.1
# pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
# # Install
# pip install matplotlib
# pip install h5py
# pip install opencv-python
# pip install pandas
# pip install transforms3d
# pip install open3d
# pip install tensorboardX

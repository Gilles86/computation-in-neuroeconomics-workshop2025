## Cloning this repository

To clone this repository locally or remotely, use the following commands in your terminal:

```bash
# Recommended location
cd $HOME/git

# Clone via HTTPS
git clone https://github.com/Gilles86/computation-in-neuroeconomics-workshop2025.git

# Or clone via SSH (if you have SSH keys set up)
git clone git@github.com:Gilles86/computation-in-neuroeconomics-workshop2025.git
```

This guide assumes you place the repository in `$HOME/git/computation-in-neuroeconomics-workshop2025`.
# Computation in Neuroeconomics Workshop 2025

Welcome to the workshop on computational techniques in Neuroeconomics 2025!  
In this repository you will find all the information, slides, code, and environments you need.

## Install environment

### Why install the Conda environments?

This workshop provides three Conda environment files to ensure everyone can run the code smoothly, regardless of setup:

- **`environment.yml`** — Standard environment for most users (Windows, Linux, Intel-based Macs).  
- **`environment_metal.yml`** — Optimized for Apple Silicon (M1/M2/M3), using Metal for GPU acceleration.  
- **`environment_gpu.yml`** — For use on the UZH ScienceCluster (with CUDA + GPUs).

---

## Installation on your own machine

1. **Install Conda or Mamba**  
   - Download [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) (recommended) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  

2. **Navigate to the `environments` folder**:
   ```bash
   cd environments
   ```

3. **Create the environment**:
   - For **Windows, Linux, Intel Macs**:
     ```bash
     conda env create -f environment.yml
     ```
   - For **Apple Silicon Macs**:
     ```bash
     conda env create -f environment_metal.yml
     ```

4. **Activate the environment**:
   ```bash
   conda activate neuroeconomics_2025
   ```
   *(Replace `neuroeconomics_2025` with the actual name from the `.yml` file if different.)*

5. **Optional: verify GPU support**:
   ```bash
   python check_for_gpu.py
   ```

---

## Installation on the UZH ScienceCluster

On the ScienceCluster, you **must not install environments in `$HOME`** (limited quota).  
Always use `/data/$USER` for Conda/Mamba and environments.  
See also the official docs: [How to use Conda on the ScienceCluster](https://docs.s3it.uzh.ch/how-to_articles/how_to_use_conda/).

### Step 1: Load Mamba (recommended)
```bash
module load mamba
```

Configure Conda to always place envs/packages in `/data/$USER` by creating `~/.condarc`:
```yaml
envs_dirs:
  - /data/$USER/conda/envs
pkgs_dirs:
  - /data/$USER/conda/pkgs
```

*(Alternatively, you can install your own Mambaforge under `/data/$USER/mambaforge`, but `module load mamba` is preferred.)*

### Step 2: Open a GPU node
TensorFlow must be installed on a GPU node so it links correctly to CUDA libraries:
```bash
srun --gres=gpu:1 --time=60:00 --mem=32G --cpus-per-task=8 --pty bash
```

Then load mamba:
```bash
module load mamba
```

### Step 3: Load CUDA drivers
```bash
module load gpu
```

### Step 4: Create the GPU environment
Navigate to the repo’s `environments` folder and create the environment:
```bash
cd /$HOME/git/computation-in-neuroeconomics-workshop2025/environments # replace with whatever you use
mamba env create -f environment_gpu.yml
```

### Step 5: Activate
```bash
conda activate soglio_cuda
```

### Step 6: Check for GPU
You can use the script `check_for_gpu.py` to check whether Tensorflow can find you GPU:

```bash
python check_for_gpu.py
```

Which should output something like
```bash
2025-09-12 12:19:11.012999: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-09-12 12:19:11.047671: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-09-12 12:19:11.047719: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2025-09-12 12:19:11.049371: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-09-12 12:19:11.055219: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
TensorFlow version: 2.16.2
NumPy version: 1.26.4
GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
2025-09-12 12:19:14.321458: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1928] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 31134 MB memory:  -> device: 0, name: Tesla V100-SXM2-32GB, pci bus id: 0000:89:00.0, compute capability: 7.0
DDM test successful: (1000, 100)
```

(Don't worry about the warnings about oneDNN, cuDNN, etc. This is fine. The point is that it finds `GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`)


## Slides and resources

### Slides
You can find the raw Markdown, PDF, and HTML versions of the slides in the [slides](./slides) directory.

### Further reading
- [Braincoder tutorials](https://braincoder-devs.github.io/)  
- [TensorFlow guide](https://www.tensorflow.org/guide)  


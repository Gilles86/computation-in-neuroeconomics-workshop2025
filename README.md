# Computation in Neuroeconomics Workshop 2025

Welcome to the workshop on computational techniques in Neuroeconomics 2025!  
In this repository you will find all the information, slides, code, and environments you need.

---

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

---

### Step 2: Open a GPU node
TensorFlow must be installed on a GPU node so it links correctly to CUDA libraries:
```bash
srun --gres=gpu:1 --time=60:00 --mem=32G --cpus-per-task=8 --pty bash
```

---

### Step 3: Load CUDA drivers
```bash
module load gpu
module load cuda/11.8   # adjust version if needed
```

---

### Step 4: Create the GPU environment
Navigate to the repo’s `environments` folder and create the environment:
```bash
cd /data/$USER/computation-in-neuroeconomics-workshop2025/environments
mamba env create -f environment_gpu.yml
```

---

### Step 5: Activate
```bash
conda activate soglio_cuda
```

---

## Slides and resources

### Slides
You can find the raw Markdown, PDF, and HTML versions of the slides in the [slides](./slides) directory.

### Further reading
- [Braincoder tutorials](https://braincoder-devs.github.io/)  
- [TensorFlow guide](https://www.tensorflow.org/guide)  


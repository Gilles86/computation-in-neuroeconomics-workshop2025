# Computation in Neuroeconomics Workshop 2025

Welcome to the workshop on computational techniques in Neuroeconomics 2025! In this github repo you will find all the information, slides, code, environments you need!

## Install environment
### Why Install the Conda Environments?

This workshop provides two Conda environment files to ensure everyone can run the code smoothly, regardless of their hardware:

- **`environment.yml`**: The standard environment for most users (Windows, Linux, or Intel-based Macs).
- **`environment_metal.yml`**: Specifically optimized for MacBooks with Apple Silicon (M1/M2/M3 chips). This environment leverages the Metal framework for accelerated computation, which is native to Apple’s ARM architecture.

### How to Install the Conda Environments

1. **Install Conda**: If you don’t have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).

2. **Navigate to the `environments` folder**:
   ```bash
   cd environments
   ```

3. **Create the environment**:
   - For **Intel-based Macs, Windows, or Linux**:
     ```bash
     conda env create -f environment.yml
     ```
   - For **MacBooks with M1/M2/M3 chips**:
     ```bash
     conda env create -f environment_metal.yml
     ```

4. **Activate the environment**:
   ```bash
   conda activate neuroeconomics_2025
   ```
   *(Replace `neuroeconomics_2025` with the actual environment name specified in the `.yml` file, if different.)*

5. **Verify the installation** (optional):
   Run the provided `check_for_gpu.py` script to ensure your environment is correctly set up and detects your hardware:
   ```bash
   python check_for_gpu.py
   ```
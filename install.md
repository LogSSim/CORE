# Installation

The original development environment used an NVIDIA 4090 GPU, CUDA 11.8, and
MuJoCo 2.1.0. Adjust package versions if your CUDA or driver version differs.

## 1. Create Environment

```bash
conda create -n CORE python=3.8
conda activate CORE
```

## 2. Install PyTorch

For CUDA 11.8:

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

Use the PyTorch build that matches your local CUDA runtime.

## 3. Install CORE

```bash
cd CORE
pip install -e .
cd ..
```

The editable package and source directory are both named `core`, which matches
the internal imports and Hydra targets.

## 4. Install MuJoCo

```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
  -O mujoco210.tar.gz --no-check-certificate
tar -xvzf mujoco210.tar.gz
```

Add the following to your shell startup file, for example `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl
```

Then reload the shell:

```bash
source ~/.bashrc
```

## 5. Install mujoco-py

```bash
cd third_party/mujoco-py-2.1.2.14
pip install -e .
cd ../..
```

## 6. Install Simulation Environments

```bash
pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

cd third_party
cd gym-0.21.0 && pip install -e . && cd ..
cd Metaworld && pip install -e . && cd ..
cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ..
cd ..
```

## 7. Install PyTorch3D

Use either a prebuilt package matching your CUDA/PyTorch version, or the bundled
simplified version:

```bash
cd third_party/pytorch3d_simplified
pip install -e .
cd ../..
```

## 8. Install Python Dependencies

```bash
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf \
  hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 \
  numba==0.56.4 moviepy imageio av matplotlib termcolor natsort open3d \
  swanlab
```

If your environment requires newer transformer/diffusers utilities:

```bash
pip install --upgrade diffusers transformers huggingface_hub
```

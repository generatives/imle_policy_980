#!/bin/bash
###############################################
# 1. Make Conda available in batch scripts
###############################################
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -d ~/miniconda3/bin ]; then
    # fallback if conda.sh missing but dir exists
    export PATH=~/miniconda3/bin:$PATH
else
    echo "Miniconda not found. Installing..."
    mkdir -p ~/miniconda3
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/etc/profile.d/conda.sh
fi

###############################################
# 2. Create env only if missing
###############################################
ENV_NAME="imle_policy"

if conda info --envs | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME'..."
    export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
    conda create -y -n "$ENV_NAME" -c conda-forge \
        python=3.10 evdev=1.9.0 \
        xorg-x11-proto-devel-cos6-x86_64 \
        glew mesa-libgl-devel-cos6-x86_64 libglib
fi

conda activate "$ENV_NAME"

###############################################
# 3. Install your package in editable mode
#    (only if not already installed)
###############################################
if ! pip show imle_policy >/dev/null 2>&1; then
    echo "Installing imle_policy (pip install -e .)..."
    pip install -e .
else
    echo "imle_policy already installed."
fi

###############################################
# 4. Run your training job
###############################################
export WANDB_API_KEY="3edf01a34993112b3c0a356b0280f938b54e2247"

CONFIG="$1"
echo "Running training with config: $CONFIG"

python -m torch.distributed.run \
    --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:0\
    imle_policy/train.py \
    --config "$CONFIG"
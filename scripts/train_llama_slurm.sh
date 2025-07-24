#!/bin/bash

#SBATCH -p preempt
#SBATCH -J llama_slurm
#SBATCH -N 1
#SBATCH -t 0-24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=256GB
#SBATCH --mail-type=END
#SBATCH --mail-user=tevinw@andrew.cmu.edu
#SBATCH --overcommit
#SBATCH --output=out.log
#SBATCH --exclude=babel-13-1,babel-9-3,babel-9-7,babel-9-11,babel-13-13,babel-13-17,babel-14-5,babel-6-21,babel-4-21,babel-13-5,babel-8-9,babel-4-1,babel-12-9,babel-14-1
#SBATCH --requeue

# Define the training script and GPU configuration
readonly training_script="train_sft_llama_ultrafeedback.sh" 
readonly GPUS_PER_NODE=8

# Set up project and container image paths
readonly PROJECT_PATH=$(cd ../src/OpenRLHF; pwd)
readonly IMAGE="docker://nvcr.io/nvidia/pytorch:24.07-py3"  # Base image from NVIDIA
readonly JOBLOG="$(pwd)/logs/$training_script-$SLURM_JOB_ID.log"
mkdir -p logs

# Log job start time
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# Source the training script to load the training commands
source ./${training_script} slurm
echo "training_commands:" &>> ${JOBLOG}
echo $training_commands &>> ${JOBLOG}

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export HF_HOME=/root/huggingface
export APPTAINER_CACHEDIR=/data/user_data/tevinw/.apptainer/cache

# On first run, install ucx and ucc libraries
git clone https://github.com/openucx/ucx || true; \
cd ucx; \
./autogen.sh; \
./configure --prefix=/root/local; \
make -j install; \
cd ..; \
git clone https://github.com/openucx/ucc || true; \
cd ucc; \
./autogen.sh; \
./configure --prefix=/root/local --with-ucx=/root/local; \
make -j install; \

# Run the training job using Apptainer with fakeroot and writable tmpfs
srun apptainer exec --nv --fakeroot --writable-tmpfs \
    --bind "$PROJECT_PATH:/openrlhf,/data/user_data/tevinw/openrlhf/checkpoint:/root/checkpoint,/data/group_data/cx_group/RuleRLHF/checkpoint/:/root/checkpoint2,/data/user_data/tevinw/local:/root/local,/data/user_data/tevinw/.cache:/root/.cache,/data/hf_cache:/root/huggingface" \
    $IMAGE \
    bash -c "cd /openrlhf; \
              apt-get update && apt-get install -y libopenmpi-dev git build-essential cmake; \
              export LD_LIBRARY_PATH=/root/local/lib:\$LD_LIBRARY_PATH; \
              export NCCL_ASYNC_ERROR_HANDLING=1; \
              export NCCL_P2P_DISABLE=1; \
              export NCCL_IB_DISABLE=1; \
              export SSL_CERT_FILE=/openrlhf/cacert.pem; \
              cd /openrlhf; \
              pip install --upgrade pip wheel setuptools; \
              MAX_JOBS=32 pip install .  --no-build-isolation --cache-dir /root/.cache/pip; \
              torchrun --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
              --master_addr $MASTER_ADDR --master_port $MASTER_PORT -m ${training_commands}" &>> ${JOBLOG}

# Log job end time
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}

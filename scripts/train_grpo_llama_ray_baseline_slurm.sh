#!/bin/bash

#SBATCH -p general        
#SBATCH -J llama_grpo_baseline
#SBATCH -N 1
#SBATCH -t 0-24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512GB
#SBATCH --mail-type=END
#SBATCH --mail-user=tevinw@andrew.cmu.edu
#SBATCH --overcommit
#SBATCH --output=out.log
#SBATCH --exclude=babel-13-1,babel-9-3,babel-9-7,babel-9-11,babel-13-13,babel-13-17,babel-14-5,babel-6-21,babel-4-21,babel-13-5,babel-8-9,babel-4-1,babel-12-9,babel-14-1

# Project settings
OPENRLHF_PATH=$(cd ../src/OpenRLHF; pwd)
# Bind the project directory and local cache directory into the container
BIND_MOUNTS="$OPENRLHF_PATH:/openrlhf,/data/group_data/cx_group/RuleRLHF/checkpoint/:/root/checkpoint2,/data/user_data/tevinw/openrlhf/checkpoint:/root/checkpoint,/data/user_data/tevinw/local:/root/local,/data/user_data/tevinw/.cache:/root/.cache,/data/user_data/tevinw/huggingface:/root/huggingface"
# Use the docker URI for Apptainer
IMAGE_NAME="docker://nvcr.io/nvidia/pytorch:24.07-py3"
RAY_VERSION=2.44.0

JOBLOG="$(realpath .)/train_ppo_llama_ray-$SLURM_JOB_ID.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# Launch ray daemon (head)
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")  # Get node names
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1

port=6379
ip_head=$ip:$port
export ip_head
export HF_HOME=/root/huggingface
export APPTAINER_CACHEDIR=/data/user_data/tevinw/.apptainer/cache

echo "IP Head: $ip_head"  &>> ${JOBLOG}

echo "STARTING HEAD at $node_1"  &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$node_1" \
    apptainer exec --nv --fakeroot --writable-tmpfs --bind "$BIND_MOUNTS" $IMAGE_NAME \
    bash -c "apt-get update && apt-get install -y libopenmpi-dev git build-essential cmake; \
             export LD_LIBRARY_PATH=/root/local/lib:\$LD_LIBRARY_PATH; \
             export NCCL_ASYNC_ERROR_HANDLING=1; \
             ls /root/checkpoint2; \
             export NCCL_P2P_DISABLE=1; \
             export NCCL_IB_DISABLE=1; \
             export VLLM_USE_V1=1; \
             export VLLM_ENABLE_V1_MULTIPROCESSING=0; \
             export RAY_TMPDIR=/root/ray_tmp; \
             mkdir -p \$RAY_TMPDIR; \
             export SSL_CERT_FILE=/openrlhf/cacert.pem; \
             pip install ray[default]==$RAY_VERSION && \
             pip show ray && \
             ray start --head --node-ip-address=$ip --port=$port --block" &>> ${JOBLOG} &
sleep 90s

# Launch ray workers on remaining nodes
worker_num=$((SLURM_JOB_NUM_NODES))
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
    srun --nodes=1 --ntasks=1 -w "$node_i" \
         apptainer exec --nv --fakeroot --writable-tmpfs --bind "$BIND_MOUNTS" $IMAGE_NAME \
         bash -c "apt-get update && apt-get install -y libopenmpi-dev git build-essential cmake; \
                  export LD_LIBRARY_PATH=/root/local/lib:\$LD_LIBRARY_PATH; \
                  export NCCL_ASYNC_ERROR_HANDLING=1; \
                  export NCCL_P2P_DISABLE=1; \
                  export NCCL_IB_DISABLE=1; \
                  export VLLM_USE_V1=1; \
                  export VLLM_ENABLE_V1_MULTIPROCESSING=0; \
                  export SSL_CERT_FILE=/openrlhf/cacert.pem; \
                  export RAY_TMPDIR=/root/ray_tmp; \
                  mkdir -p \$RAY_TMPDIR; \
                  pip install ray[default]==$RAY_VERSION && \
                  pip show ray && \
                  ray start --address $ip_head --block" &>> ${JOBLOG} &
    sleep 1s;
done

sleep 30s

# ===== Submit ray job with the new command =====
srun --overlap --nodes=1 --ntasks=1 -w "$node_1" \
    apptainer exec --nv --fakeroot --writable-tmpfs --bind "$BIND_MOUNTS" $IMAGE_NAME \
    bash -c "apt-get update && apt-get install -y libopenmpi-dev git build-essential cmake; \
             export LD_LIBRARY_PATH=/root/local/lib:\$LD_LIBRARY_PATH; \
             export NCCL_ASYNC_ERROR_HANDLING=1; \
             export NCCL_P2P_DISABLE=1; \
             export NCCL_IB_DISABLE=1; \
             export VLLM_USE_V1=1; \
             export VLLM_ENABLE_V1_MULTIPROCESSING=0; \
             export SSL_CERT_FILE=/openrlhf/cacert.pem; \
             pip install ray[default]==$RAY_VERSION && \
             pip show ray && \
             mkdir -p /tmp/ray; \
             chmod 777 /tmp/ray; \
             ray job submit --address=\"http://127.0.0.1:8265\" \
             --runtime-env-json='{\"working_dir\": \"/openrlhf\", \"pip\": \"/openrlhf/requirements.txt\"}' \
             -- python3 -m openrlhf.cli.train_ppo_ray \
                 --ref_num_nodes 1 \
                 --ref_num_gpus_per_node 1 \
                 --reward_num_nodes 1 \
                 --reward_num_gpus_per_node 1 \
                 --actor_num_nodes 1 \
                 --actor_num_gpus_per_node 4 \
                 --vllm_num_engines 2 \
                 --vllm_tensor_parallel_size 1 \
                 --enable_prefix_caching \
                 --pretrain /root/checkpoint/ar-llama-3-8b-sft-ultrafeedback \
                 --reward_pretrain /root/checkpoint/ar-llama-3-8b-rm \
                 --judge_pretrain meta-llama/Meta-Llama-3-8B-Instruct \
                 --save_path /root/checkpoint2/ar-llama-3-8b-grpo-baseline \
                 --ckpt_path /root/checkpoint2/ar-llama-3-8b-grpo-baseline-ckpt \
                 --save_hf_ckpt \
                 --max_ckpt_num 2 \
                 --save_steps 2 \
                 --micro_train_batch_size 16 \
                 --train_batch_size 128 \
                 --micro_rollout_batch_size 32 \
                 --rollout_batch_size 1024 \
                 --n_samples_per_prompt 2 \
                 --max_epochs 1 \
                 --num_episodes 1 \
                 --prompt_max_len 1024 \
                 --max_samples 100000 \
                 --generate_max_len 1024 \
                 --init_kl_coef 1e-3 \
                 --gamma 1.0 \
                 --use_kl_loss \
                 --kl_estimator k3 \
                 --advantage_estimator group_norm \
                 --zero_stage 3 \
                 --bf16 \
                 --actor_learning_rate 5e-7 \
                 --prompt_data tevinw/processed_ultrafeedback_binarized_prefs_sft \
                 --prompt_split train_prefs \
                 --input_key prompt \
                 --normalize_reward \
                 --adam_offload \
                 --ref_reward_offload \
                 --gradient_checkpointing \
                 --packing_samples \
                 --vllm_sync_backend nccl \
                 --flash_attn \
                 --use_wandb $WANDB_TOKEN" &>> ${JOBLOG}



echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}

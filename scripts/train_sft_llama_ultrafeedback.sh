set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --dataset tevinw/processed_ultrafeedback_binarized_prefs_sft \
   --train_split train_sft \
   --eval_split test_sft \
   --input_key prompt \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --lr_scheduler constant_with_warmup \
   --lr_warmup_ratio 0.05 \
   --max_norm 10.0 \
   --input_template $'<|user|>{}<|assistant|>' \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path /root/checkpoint/ar-llama-3-8b-sft-ultrafeedback \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb $WANDB_TOKEN
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

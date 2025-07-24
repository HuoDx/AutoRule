set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path /root/checkpoint/ar-llama-3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --pretrain /root/checkpoint/ar-llama-3-8b-sft-ultrafeedback \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-6 \
   --lr_scheduler constant_with_warmup \
   --lr_warmup_ratio 0.1 \
   --max_norm 5.0 \
   --dataset tevinw/processed_ultrafeedback_binarized_prefs_sft \
   --train_split train_prefs \
   --eval_split test_prefs \
   --input_template $'<|user|>{}<|assistant|>' \
   --prompt_key prompt \
   --chosen_key chosen_content \
   --rejected_key rejected_content \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb $WANDB_TOKEN
EOF


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

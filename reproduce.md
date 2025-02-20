## Training
### Training on TSP-50
CUDA_VISIBLE_DEVICES=6,7 accelerate-launch --main_process_port 29502 train_co.py \
    --run_name='cado' \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --sample_iters 16 \
    --num_epochs 3000 \
    --eval_step -20 \
    --inference_diffusion_steps 20 \
    --learning_rate 1e-5 \
    --reward_2opt 0 \
    --task tsp \
    --use_env \
    --tsp_decoder cython_merge \
    --task_size 50 \
    --task_load_size 50 \
    --reward_baseline 0 \
    --lora_rank 2 \
    --lora_range 1 \
    --last_train 1 \
    --sparse_factor -1 \
    --num_testset -1 \
    --mixed_precision fp16 \
    --print_sl_loss 0 \
    --seed 1 \
    --num_trainset -1 \
    --use_critic 0 \
    --critic_ratio 1 \
    --kl_grdy 0.00 \
    --use_weighted_critic 0


```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate-launch train_co.py \
    --run_name='cado' \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --sample_iters 16 \
    --num_epochs 3000 \
    --eval_step -20 \
    --inference_diffusion_steps 20 \
    --learning_rate 1e-5 \
    --reward_2opt 0 \
    --task tsp \
    --use_env \
    --tsp_decoder cython_merge \
    --task_size 50 \
    --task_load_size 50 \
    --reward_baseline 0 \
    --lora_rank 2 \
    --lora_range 1 \
    --last_train 1 \
    --sparse_factor -1 \
    --num_testset -1 \
    --mixed_precision fp16 \
    --print_sl_loss 0 \
    --seed 1 \
    --num_trainset -1 \
    --use_critic 0 \
    --critic_ratio 1 \
    --kl_grdy 0.00 \
    --use_weighted_critic 0
```

### Training on TSP-100
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate-launch train_co.py \
    --run_name='cado' \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --sample_iters 16 \
    --num_epochs 3000 \
    --eval_step -20 \
    --inference_diffusion_steps 20 \
    --learning_rate 1e-5 \
    --reward_2opt 0 \
    --task tsp \
    --use_env \
    --tsp_decoder cython_merge \
    --task_size 100 \
    --task_load_size 100 \
    --reward_baseline 0 \
    --lora_rank 2 \
    --lora_range 1 \
    --last_train 1 \
    --sparse_factor -1 \
    --num_testset -1 \
    --mixed_precision fp16 \
    --print_sl_loss 0 \
    --seed 1 \
    --num_trainset -1 \
    --use_critic 0 \
    --critic_ratio 1 \
    --kl_grdy 0.00 \
    --use_weighted_critic 0
```

### Training on TSP-500
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate-launch train_co.py \
    --run_name='cado' \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --sample_iters 16 \
    --num_epochs 5000 \
    --eval_step -20 \
    --inference_diffusion_steps 20 \
    --learning_rate 1e-5 \
    --reward_2opt 0 \
    --task tsp \
    --use_env \
    --tsp_decoder cython_merge \
    --task_size 500 \
    --task_load_size 500 \
    --reward_baseline 0 \
    --lora_rank 2 \
    --lora_range 1 \
    --last_train 1 \
    --sparse_factor 50 \
    --num_testset -1 \
    --mixed_precision fp16 \
    --print_sl_loss 0 \
    --seed 1 \
    --num_trainset -1 \
    --use_critic 0 \
    --critic_ratio 1 \
    --kl_grdy 0.00 \
    --use_weighted_critic 0
```

### Training on TSP-1000
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate-launch train_co.py \
    --run_name='cado' \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --sample_iters 32 \
    --num_epochs 5000 \
    --eval_step -20 \
    --inference_diffusion_steps 20 \
    --learning_rate 1e-5 \
    --reward_2opt 0 \
    --task tsp \
    --use_env \
    --tsp_decoder cython_merge \
    --task_size 1000 \
    --task_load_size 1000 \
    --reward_baseline 0 \
    --lora_rank 2 \
    --lora_range 1 \
    --last_train 1 \
    --sparse_factor 50 \
    --num_testset -1 \
    --mixed_precision fp16 \
    --print_sl_loss 0 \
    --seed 1 \
    --num_trainset -1 \
    --use_critic 0 \
    --critic_ratio 1 \
    --kl_grdy 0.00 \
    --use_weighted_critic 0
```

### TSP 10000 (Train_iter 8)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate-launch train_co.py \
    --run_name='cado' \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --sample_iters 16 \
    --num_epochs 3000 \
    --eval_step -2 \
    --inference_diffusion_steps 10 \
    --learning_rate 1e-5 \
    --reward_2opt 0 \
    --task tsp \
    --use_env \
    --tsp_decoder cython_merge \
    --task_size 10000 \
    --task_load_size 10000 \
    --reward_baseline 0 \
    --lora_rank 0 \
    --lora_range 1 \
    --last_train 2 \
    --sparse_factor 50 \
    --num_testset -1 \
    --mixed_precision fp16 \
    --print_sl_loss 0 \
    --seed 1 \
    --num_trainset -1 \
    --use_critic 0 \
    --critic_ratio 1 \
    --kl_grdy 0.00 \
    --use_weighted_critic 0 \
    --train_iters 8
```

### MIS-ER
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate-launch train_co.py \
    --run_name='cado' \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --sample_iters 32 \
    --num_epochs 1400 \
    --eval_step -15 \
    --inference_diffusion_steps 20 \
    --learning_rate 1e-5 \
    --reward_2opt 0 \
    --task mis_er \
    --reward_baseline 0 \
    --lora_rank 2 \
    --lora_range 1 \
    --last_train 1 \
    --num_testset -1 \
    --mixed_precision fp16 \
    --print_sl_loss 0 \
    --seed 1 \
    --num_trainset -1 \
    --use_critic 0 \
    --critic_ratio 1 \
    --kl_grdy 0.00 \
    --use_weighted_critic 0
```

### MIS-SAT
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate-launch train_co.py \
    --run_name='cado' \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --sample_iters 32 \
    --num_epochs 3000 \
    --eval_step -15 \
    --inference_diffusion_steps 20 \
    --learning_rate 1e-5 \
    --reward_2opt 0 \
    --task mis_sat \
    --reward_baseline 0 \
    --lora_rank 2 \
    --lora_range 1 \
    --last_train 1 \
    --num_testset -1 \
    --mixed_precision fp16 \
    --print_sl_loss 0 \
    --seed 1 \
    --num_trainset -1 \
    --use_critic 0 \
    --critic_ratio 1 \
    --use_weighted_critic 0 \
    --kl_grdy 0.0
```

## Evaluation
### TSP-50
```bash

keep_running=true

# Define SIGINT signal handler function
handle_sigint() {
    echo "Loop terminated due to Ctrl-C input."
    keep_running=false
}

# Trap SIGINT signal and bind it to the handler function
trap 'handle_sigint' SIGINT

# Define args_list array (each element is a string of 4 arguments separated by spaces)
# reward opt, rewrite, parallel_sampling,
N=${#args_list[@]}
args_list=(
    "0 0 1"
    "0 1 1"
    "1 1 1"
    "0 1 4"
    "1 1 4"
)

# Execute the command in a loop N times.
for ((i=0; i<N; i++))
do
    if [ "$keep_running" = false ]; then
        echo "Exiting loop."
        break
    fi
    # Retrieve arguments from args_list.
    args="${args_list[$i]}"

    # Split the arguments.
    read -r arg1 arg2 arg3 arg4 <<< "$args"

    # Write the actual command to execute. For example, if using mycommand:
    CUDA_VISIBLE_DEVICES=1 accelerate-launch eval_cado.py \
    --task_size 50 \
    --task_load_size 50 \
    --task tsp \
    --inference_diffusion_steps 20 \
    --tsp_decoder cython_merge \
    --sparse_factor -1 \
    --reward_2opt $arg1 \
    --lora_new 0 \
    --lora_rank 2 \
    --last_train 1 \
    --lora_range 1 \
    --inference_steps 10 \
    --rewrite $arg2 \
    --rewrite_steps 3 \
    --parallel_sampling $arg3 \
    --resume_from "~/your_storage/ckpt_cado/TSP50"
    # Print progress.
    echo "Executed command with args: $arg1 $arg2 $arg3"
done
```

### TSP-100
```bash

keep_running=true

# Define SIGINT Signal Processing Function
handle_sigint() {
    echo "Ctrl-C Loop Ends."
    keep_running=false
}

# SIGINT
trap 'handle_sigint' SIGINT

# define args_list array
# reward opt, rewrite, parallel_sampling,
N=${#args_list[@]}
args_list=(
    "1 1 4"
    "0 0 1"
    "0 1 1"
    "0 0 1"
    "0 1 4"
)

# Running N loops 
for ((i=0; i<N; i++))
do
    if [ "$keep_running" = false ]; then
        echo "Exiting loop."
        break
    fi
    # Get arguments from args_list.
    args="${args_list[$i]}"

    # Separate the arguments.
    read -r arg1 arg2 arg3 arg4 <<< "$args"

    # Write the actual command to execute. For example, if using mycommand:
    CUDA_VISIBLE_DEVICES=0 accelerate-launch eval_cado.py \
    --task_size 100 \
    --task_load_size 100 \
    --task tsp \
    --inference_diffusion_steps 20 \
    --tsp_decoder cython_merge \
    --sparse_factor -1 \
    --reward_2opt $arg1 \
    --lora_new 0 \
    --lora_rank 2 \
    --last_train 1 \
    --lora_range 1 \
    --inference_steps 10 \
    --rewrite $arg2 \
    --rewrite_steps 3 \
    --parallel_sampling $arg3 \
    --resume_from "~/your_storage/ckpt_cado/TSP100" \
    # --mixed_precision fp16
    # Print progress.
    echo "Executed command with args: $arg1 $arg2 $arg3"
done
```


### TSP-500
```bash

keep_running=true

# Define SIGINT signal handler function
handle_sigint() {
    echo "Loop terminated due to Ctrl-C input."
    keep_running=false
}

# Trap SIGINT signal and bind it to the handler function
trap 'handle_sigint' SIGINT

# Define args_list array (each element is a string of 4 arguments separated by spaces)
# reward opt, rewrite, parallel_sampling,
N=${#args_list[@]}
args_list=(
    "0 0 1"
    "0 1 1"
    "0 1 4"
    "1 1 1"
    "1 1 4"
)

# Execute the command in a loop N times.
for ((i=0; i<N; i++))
do
    if [ "$keep_running" = false ]; then
        echo "Exiting loop."
        break
    fi
    # Retrieve arguments from args_list.
    args="${args_list[$i]}"

    # Split the arguments.
    read -r arg1 arg2 arg3 arg4 <<< "$args"

    # Write the actual command to execute. For example, if using mycommand:
    CUDA_VISIBLE_DEVICES=0 accelerate-launch eval_cado.py \
    --task_size 500 \
    --task_load_size 500 \
    --task tsp \
    --inference_diffusion_steps 20 \
    --tsp_decoder cython_merge \
    --sparse_factor 50 \
    --reward_2opt $arg1 \
    --lora_new 1 \
    --lora_rank 2 \
    --last_train 1 \
    --lora_range 1 \
    --inference_steps 10 \
    --rewrite $arg2 \
    --rewrite_steps 3 \
    --parallel_sampling $arg3 \
    --mixed_precision fp16 \
    --resume_from "~/your_storage/ckpt_cado/TSP500" \
    echo "Executed command with args: $arg1 $arg2 $arg3"
done
```


### TSP-1000
```bash
keep_running=true

# Define SIGINT signal handler function
handle_sigint() {
    echo "Loop terminated due to Ctrl-C input."
    keep_running=false
}

# Trap SIGINT signal and bind it to the handler function
trap 'handle_sigint' SIGINT

# Define args_list array (each element is a string of 4 arguments separated by spaces)
# reward opt, rewrite, parallel_sampling,
N=${#args_list[@]}
args_list=(
    "0 0 1"
    "0 1 1"
    "1 1 1"
    "0 1 4"
    "1 1 4"
)

# Execute the command in a loop N times.
for ((i=0; i<N; i++))
do
    if [ "$keep_running" = false ]; then
        echo "Exiting loop."
        break
    fi
    # Retrieve arguments from args_list.
    args="${args_list[$i]}"

    # Split the arguments.
    read -r arg1 arg2 arg3 arg4 <<< "$args"

    # Write the actual command to execute. For example, if using mycommand:
    CUDA_VISIBLE_DEVICES=0 accelerate-launch --main_process_port 29503  eval_cado.py \
    --task_size 1000 \
    --task_load_size 1000 \
    --task tsp \
    --inference_diffusion_steps 20 \
    --tsp_decoder cython_merge \
    --sparse_factor 50 \
    --reward_2opt $arg1 \
    --lora_new 0 \
    --lora_rank 2 \
    --last_train 1 \
    --lora_range 1 \
    --inference_steps 10 \
    --rewrite $arg2 \
    --rewrite_steps 3 \
    --parallel_sampling $arg3 \
    --resume_from "~/your_storage/ckpt_cado/TSP1000" \
    --mixed_precision fp16 \
    # Print progress.
    echo "Executed command with args: $arg1 $arg2 $arg3"
done
```


### TSP-10000
```bash
keep_running=true

# Define SIGINT signal handler function
handle_sigint() {
    echo "Terminating loop due to Ctrl-C input."
    keep_running=false
}

# Trap SIGINT signal and bind it to the handler function
trap 'handle_sigint' SIGINT

# Define args_list array (each element is a string of 4 arguments separated by spaces)
# reward opt, rewrite, parallel_sampling,
N=${#args_list[@]}
args_list=(
    "0 0 1"
    "1 0 1"
    "0 0 4"
    "1 0 4"
)

# Execute the command in a loop N times.
for ((i=0; i<N; i++))
do
    if [ "$keep_running" = false ]; then
        echo "Exiting loop."
        break
    fi
    # Retrieve arguments from args_list.
    args="${args_list[$i]}"

    # Split the arguments.
    read -r arg1 arg2 arg3 arg4 <<< "$args"

    # Write the actual command to execute. For example, if using mycommand:
    CUDA_VISIBLE_DEVICES=0 accelerate-launch eval_cado.py \
    --task_size 10000 \
    --task_load_size 10000 \
    --task tsp \
    --inference_diffusion_steps 20 \
    --tsp_decoder cython_merge \
    --sparse_factor 50 \
    --reward_2opt $arg1 \
    --lora_new 1 \
    --lora_rank 0 \
    --last_train 1 \
    --lora_range 1 \
    --inference_steps 10 \
    --rewrite $arg2 \
    --rewrite_steps 3 \
    --parallel_sampling $arg3 \
    --use_env \
    --mixed_precision fp16 \
    --resume_from "~/your_storage/ckpt_cado/TSP10000" \
    # Print progress.
    echo "Executed command with args: $arg1 $arg2 $arg3"
done
```

### MIS-ER
```bash
keep_running=true

# Define SIGINT signal handler function
handle_sigint() {
    echo "Loop terminated due to Ctrl-C input."
    keep_running=false
}

# Trap SIGINT signal and attach the handler function
trap 'handle_sigint' SIGINT

# Define args_list array (each element is a string with 4 arguments separated by spaces)
# reward opt, rewrite, parallel_sampling,
N=${#args_list[@]}
args_list=(
    "0 0 1"
    "0 1 1"
    "1 1 4"
)

# Execute the command in a loop N times.
for ((i=0; i<N; i++))
do
    if [ "$keep_running" = false ]; then
        echo "Exiting loop."
        break
    fi

    # Retrieve arguments from args_list.
    args="${args_list[$i]}"

    # Split the arguments.
    read -r arg1 arg2 arg3 arg4 <<< "$args"

    # Write the actual command to execute. For example, if using mycommand:
    CUDA_VISIBLE_DEVICES=0 accelerate-launch --main_process_port 29502 eval_cado.py \
    --task_size 100 \
    --task_load_size 100 \
    --task mis_er \
    --inference_diffusion_steps 20 \
    --tsp_decoder cython_merge \
    --sparse_factor -1 \
    --reward_2opt $arg1 \
    --lora_new 0 \
    --lora_rank 2 \
    --last_train 1 \
    --lora_range 1 \
    --inference_steps 10 \
    --rewrite $arg2 \
    --rewrite_steps 3 \
    --parallel_sampling $arg3 \
    --rewrite_ratio 0.1 \
    --resume_from "~/your_storage/ckpt_cado/MIS_ER" \
    --seed 1
    # --mixed_precision fp16 \
    # Print progress.
    echo "Executed command with args: $arg1 $arg2 $arg3"
done
```


### MIS-SAT
```bash

keep_running=true

# Define SIGINT signal handler function
handle_sigint() {
    echo "Loop terminated due to Ctrl-C input."
    keep_running=false
}

# Trap SIGINT signal and bind it to the handler function
trap 'handle_sigint' SIGINT

# Define args_list array (each element is a string of 4 arguments separated by spaces)
# reward opt, rewrite, parallel_sampling,
N=${#args_list[@]}
args_list=(
    "0 0 1"
    "0 1 1"
    "0 1 4"
)

# Execute the command in a loop N times.
for ((i=0; i<N; i++))
do
    if [ "$keep_running" = false ]; then
        echo "Exiting loop."
        break
    fi
    # Retrieve arguments from args_list.
    args="${args_list[$i]}"

    # Split the arguments.
    read -r arg1 arg2 arg3 arg4 <<< "$args"

    # Write the actual command to execute. For example, if using mycommand:
    CUDA_VISIBLE_DEVICES=1 accelerate-launch --main_process_port 29503  eval_cado.py \
    --task_size 100 \
    --task_load_size 100 \
    --task mis_sat \
    --inference_diffusion_steps 20 \
    --tsp_decoder cython_merge \
    --sparse_factor -1 \
    --reward_2opt $arg1 \
    --lora_new 1 \
    --lora_rank 2 \
    --last_train 1 \
    --lora_range 1 \
    --inference_steps 10 \
    --rewrite $arg2 \
    --rewrite_steps 3 \
    --parallel_sampling $arg3 \
    --rewrite_ratio 0.1 \
    --resume_from "~/your_storage/ckpt_cado/MIS_SAT" \
    # --mixed_precision fp16 \
    # Print progress.
    echo "Executed command with args: $arg1 $arg2 $arg3"
done
```

### TSP-Lib (50-200)
```bash
#!/bin/bash

# Count execution

keep_running=true

# Define SIGINT Signal Processing Function
handle_sigint() {
    echo "Ctrl-C Loop Ends."
    keep_running=false
}

# SIGINT
trap 'handle_sigint' SIGINT

# define args_list array
# reward opt, rewrite, parallel_sampling,
N=${#args_list[@]}
args_list=(
    "0 1 1"
)

# Running N loops 
for ((i=0; i<N; i++))
do
    if [ "$keep_running" = false ]; then
        echo "Exiting loop."
        break
    fi
    # Get arguments from args_list.
    args="${args_list[$i]}"

    # Separate the arguments.
    read -r arg1 arg2 arg3 arg4 <<< "$args"

    # Write the actual command to execute. For example, if using mycommand:
    CUDA_VISIBLE_DEVICES=0 accelerate-launch eval_cado.py \
    --task_size 100 \
    --task_load_size 100 \
    --task tsp \
    --inference_diffusion_steps 20 \
    --tsp_decoder cython_merge \
    --sparse_factor -1 \
    --reward_2opt $arg1 \
    --lora_new 0 \
    --lora_rank 2 \
    --last_train 1 \
    --lora_range 1 \
    --inference_steps 10 \
    --rewrite $arg2 \
    --rewrite_steps 3 \
    --parallel_sampling $arg3 \
    --resume_from "~/your_storage/ckpt_cado/TSP100" \
    --tsplib 1 \
    # --mixed_precision fp16
    # Print progress.
    echo "Executed command with args: $arg1 $arg2 $arg3"
done
```


### TSP-Lib (200-1000)
```bash

keep_running=true

# Define SIGINT signal handler function
handle_sigint() {
    echo "Loop terminated due to Ctrl-C input."
    keep_running=false
}

# Trap SIGINT signal and bind it to the handler function
trap 'handle_sigint' SIGINT

# Define args_list array (each element is a string of 4 arguments separated by spaces)
# reward opt, rewrite, parallel_sampling,
N=${#args_list[@]}
args_list=(
    "0 0 1"
    "0 1 1"
    "0 1 4"
    "1 1 1"
    "1 1 4"
)

# Execute the command in a loop N times.
for ((i=0; i<N; i++))
do
    if [ "$keep_running" = false ]; then
        echo "Exiting loop."
        break
    fi
    # Retrieve arguments from args_list.
    args="${args_list[$i]}"

    # Split the arguments.
    read -r arg1 arg2 arg3 arg4 <<< "$args"

    # Write the actual command to execute. For example, if using mycommand:
    CUDA_VISIBLE_DEVICES=0 accelerate-launch eval_cado.py \
    --task_size 500 \
    --task_load_size 500 \
    --task tsp \
    --inference_diffusion_steps 20 \
    --tsp_decoder cython_merge \
    --sparse_factor 50 \
    --reward_2opt $arg1 \
    --lora_new 1 \
    --lora_rank 2 \
    --last_train 1 \
    --lora_range 1 \
    --inference_steps 10 \
    --rewrite $arg2 \
    --rewrite_steps 3 \
    --parallel_sampling $arg3 \
    --mixed_precision fp16 \
    --tsplib 1 \
    --resume_from "~/your_storage/ckpt_cado/TSP500" \
    echo "Executed command with args: $arg1 $arg2 $arg3"
done
```
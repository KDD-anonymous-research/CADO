import contextlib
from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from argparse import ArgumentParser
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
# from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob 
from ddpo_pytorch.diffusers_patch.difusco_logprob import difusco_with_logprob,categorical_denoise_step, difusco_with_logprob_mis, categorical_denoise_step_mis, gaussian_denoise_step_mis
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch.nn as nn
from difusco.models.gnn_encoder import AddLora
import copy as cp
import torch.nn.functional as F
# from difusco.models.gnn_encoder import LoraGNN
# from difusco.models.gnn_encoder import AddLoraLayer, LoraLayer

from accelerate import DistributedDataParallelKwargs
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
# from config.difsuco_args import arg_parser
from difusco.utils.diffusion_schedulers import InferenceSchedule
from difusco.utils.tsp_utils import calculate_distance_matrix
from difusco.co_datasets.tsp_graph_dataset import TSPGraphEnvironment
from difusco.co_datasets.mis_dataset import MIS_ERGraphEnvironment
import copy

import os

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base_co.py", "Training configuration.")

flags.DEFINE_bool('use_sweep', False, 'Whether to use Weights & Biases sweep for hyperparameter search.')
flags.DEFINE_string('run_name', "", 'Your name')
flags.DEFINE_string('task', "tsp", 'tsp, mis_sat, mis_er')
flags.DEFINE_integer('task_sweep', 0, 'sweep to mis_sat vs mis_er')
flags.DEFINE_integer('task_size', 100, "task size to solve")
flags.DEFINE_integer('task_load_size', 100, "the task size that should be loaded ckpt")

flags.DEFINE_integer('batch_size', 4, "a") # Evaluation이나 처음 sampling을 할 때 사용하는 세팅
flags.DEFINE_integer('gradient_accumulation_steps',1,"a") #100의 약수
flags.DEFINE_integer('sample_iters', 2,"a")
flags.DEFINE_integer('train_iters', 1, "a")
flags.DEFINE_integer('num_epochs', 100,"a")
flags.DEFINE_integer('print_sl_loss', 1,"a")

flags.DEFINE_integer('eval_step', -5,"number of steps to evaluate, -1: evaluation for each epoch, 1000: evaluateion whenever for 1000 global steps")

flags.DEFINE_integer('num_workers', 1,"a")
flags.DEFINE_integer('inference_diffusion_steps', 20, "a")

flags.DEFINE_string('tsp_decoder', 'cython_merge', "cython_merge or am_decoding or farthest")

flags.DEFINE_integer('critic_tstart', 0, "0: baseline for divided by all time step, 1: basline for not all time ")
flags.DEFINE_integer('sparse_factor', -1, "a")
flags.DEFINE_integer('two_opt_iterations', 1000, "a")
flags.DEFINE_integer('reward_2opt', 0, "1: with 2opt, 0: without 2opt")
flags.DEFINE_bool('reward_gap', False, "False: do not use reward gap, 1: use reward gap")   
flags.DEFINE_integer('reward_baseline', 0, "0: normal reward, 1: reward with self baseline, 2: reward with optimal baseline")
flags.DEFINE_integer('roll_out_num', 32, "Num of roll outs for reward baseline")

flags.DEFINE_string('resume_from', "", 'resume from checkpoint')
flags.DEFINE_string('subopt', "", "name for reading")
flags.DEFINE_integer("parallel_sampling", 1, "parallel sampling")
flags.DEFINE_bool("use_activation_checkpoint", False, "use activation checkpoint")

# flags.DEFINE_bool("lora", True, "use lora")
flags.DEFINE_string("mixed_precision", 'fp16', "check mixed precision, no or fp16")
flags.DEFINE_float('learning_rate', 1e-5, "learning rate")
flags.DEFINE_float('learning_rate_aux', 2e-4, "learning rate")
flags.DEFINE_bool('use_env', False, 'generate datasamples during training')
# flags.DEFINE_bool('use_env', False, 'generate datasamples during training')

flags.DEFINE_integer('last_train', 1, 'train only last layer')
flags.DEFINE_integer("lora_rank", 2, "use lora")
flags.DEFINE_integer("lora_new", 1, "0: old lora, 1: new lora")
flags.DEFINE_integer("lora_range", 1, "use lora")
flags.DEFINE_integer('num_testset', 128, 'number of testset')
flags.DEFINE_integer('num_trainset', -1, 'number of trainset')

flags.DEFINE_integer("use_critic",0, "use critic: 0: REINFORCE, 1: critic for decoded sol, 2: value only")
flags.DEFINE_integer("use_weighted_critic", 0, "0: just kl, 1: weighted critic")


flags.DEFINE_integer('seed', -1, 'seed for random number generator, -1: random seed')
flags.DEFINE_bool('reward_shaping', False, 'use reward shaping')
flags.DEFINE_float('kl_grdy', 0.0, 'kl divergence between greedy decoded output')
flags.DEFINE_float('decoded_penalty', 0.0, 'ratio of penalty when the decoded sol is changed from latent')

flags.DEFINE_float('kl_pretrain', 0.00, 'kl divergence between original difusco output')
flags.DEFINE_float('kl_aux', 0.00, 'kl divergence between auxiliary')
flags.DEFINE_float('critic_ratio', 1., 'critic learning rate ratio compared to actor')


    # learning rate.
logger = get_logger(__name__)
# torch.use_deterministic_algorithms(True)
def classify_parameters(model):
    params_with_grad = []
    params_without_grad = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            params_with_grad.append((name, param))
        else:
            params_without_grad.append((name, param))
    
    return params_with_grad, params_without_grad


def get_batch(dataloader):
    if not hasattr(get_batch, "_iterator"):
        get_batch._iterator = iter(dataloader)
    
    try:
        batch = next(get_batch._iterator)
    except StopIteration:
        get_batch._iterator = iter(dataloader)
        batch = next(get_batch._iterator)
    
    return batch

def update_config(config,input_var):
    config.unlock() 
    config.run_name = input_var.run_name
    config.task = input_var.task
    config.task_size = input_var.task_size
    config.task_load_size = input_var.task_load_size
    config.tsp_decoder = input_var.tsp_decoder
    config.batch_size = input_var.batch_size
    config.subopt = input_var.subopt
    config.resume_from = input_var.resume_from
    config.num_workers = input_var.num_workers
    config.inference_diffusion_steps = input_var.inference_diffusion_steps
    config.sparse_factor = input_var.sparse_factor
    config.two_opt_iterations = input_var.two_opt_iterations
    config.reward_2opt = input_var.reward_2opt
    config.learning_rate = input_var.learning_rate
    config.learning_rate_aux = input_var.learning_rate_aux
    config.use_env = input_var.use_env
    config.seed = input_var.seed
    config.lora_rank = input_var.lora_rank
    config.num_epochs = input_var.num_epochs
    config.last_train = input_var.last_train
    config.num_testset = input_var.num_testset
    config.num_trainset = input_var.num_trainset
    config.lora_new = input_var.lora_new

    config.sample.num_iters_per_epoch = input_var.sample_iters
    config.train.num_inner_epochs = input_var.train_iters
    config.eval_step = input_var.eval_step
    config.reward_gap = input_var.reward_gap
    config.use_sweep = input_var.use_sweep
    config.print_sl_loss = input_var.print_sl_loss
    config.lora_range = input_var.lora_range
    config.train.gradient_accumulation_steps = input_var.gradient_accumulation_steps
    config.mixed_precision = input_var.mixed_precision
    config.reward_shaping = input_var.reward_shaping
    config.parallel_sampling = input_var.parallel_sampling
    config.use_activation_checkpoint = input_var.use_activation_checkpoint
    config.kl_pretrain = input_var.kl_pretrain
    config.kl_grdy = input_var.kl_grdy
    config.decoded_penalty = input_var.decoded_penalty
    config.kl_aux = input_var.kl_aux
    config.reward_baseline = input_var.reward_baseline
    config.roll_out_num = input_var.roll_out_num
    config.use_critic = input_var.use_critic
    config.critic_ratio = input_var.critic_ratio
    config.use_weighted_critic = input_var.use_weighted_critic
    config.critic_tstart = input_var.critic_tstart
    config.task_sweep = input_var.task_sweep

    config.run_name = f'{input_var.run_name}{input_var.task}{input_var.task_size}_steps{input_var.inference_diffusion_steps}_bs{config.batch_size},st{input_var.sample_iters},ti{input_var.train_iters},accu{config.train.gradient_accumulation_steps},lr{input_var.learning_rate}'
    
    num_processes = max(torch.cuda.device_count(), 1)

    config.run_name += f'_tb{num_processes*config.batch_size*config.train.gradient_accumulation_steps}'

    # if config.use_env :
        # config.run_name += "_useenv"
    if config.last_train>0 :
        config.run_name += f"_last{config.last_train}"
    if config.lora_rank > 0:
        config.run_name += f"_lora{config.lora_rank}"
    if config.lora_rank <= 0 and config.last_train <= 0:
        config.run_name += "_full"
    if config.kl_grdy > 0:
        config.run_name += f"_klg{config.kl_grdy}"
    if config.decoded_penalty > 0:
        config.run_name += f"_dp{config.decoded_penalty}"
    if config.kl_pretrain > 0:
        config.run_name += f"_klp{config.kl_pretrain}"
    if config.kl_aux > 0:
        config.run_name += f"_aux{config.kl_aux}"
    if config.seed>0:
        config.run_name += f"_sd{config.seed}"

    if config.task=="tsp":
        if config.subopt:
            config.ckpt_path = os.path.join(config.storage_path,"checkpoints",f'{config.subopt}.ckpt')
        else:
            config.ckpt_path = os.path.join(config.storage_path,"checkpoints",f'{config.task}{config.task_load_size}.ckpt')
        config.training_split = os.path.join(f'data/tsp_custom/',f'{config.task}{config.task_size}_train_{config.data_dist}.txt')
        config.test_split = os.path.join(f'data/tsp_custom/',f'{config.task}{config.task_size}_test_{config.data_dist}.txt')
    elif config.task=="mis_sat":
        assert config.diffusion_type=="categorical"
        if config.task_sweep:
            task_ckpt = "mis_er"
            config.hidden_dim = 128
        else:
            task_ckpt = config.task
        print('load task',task_ckpt)
        config.ckpt_path = os.path.join(config.storage_path,"checkpoints",f'{task_ckpt}_{config.diffusion_type}.ckpt')
        config.training_split = os.path.join(f'data/MIS_SAT_train/','*gpickle')

        config.test_split = os.path.join(f'data/MIS_SAT_test/','*gpickle') ## actually validation split

    elif config.task=="mis_er":
        # config.diffusion_type="gaussian"
        config.diffusion_type="categorical"
        if config.diffusion_type=="categorical":
            config.hidden_dim = 128
            print("config.hidden_dim",config.hidden_dim)
        if config.task_sweep:
            task_ckpt = "mis_sat"
            config.hidden_dim = 256
        else:
            task_ckpt = config.task
        print('load task',task_ckpt)
        config.ckpt_path = os.path.join(config.storage_path,"checkpoints",f'{task_ckpt}_{config.diffusion_type}.ckpt')
        config.training_split = os.path.join(f'data/MIS_ER/er_train','*gpickle')
        config.test_split = os.path.join(f'data/MIS_ER/er_test','*gpickle') ## 
        
    elif config.task=='pctsp':
        if config.subopt:
            config.ckpt_path = os.path.join(config.storage_path,"checkpoints",f'{config.subopt}.ckpt')
        else:
            config.ckpt_path = os.path.join(config.storage_path,"checkpoints",'tsp'+f'{config.task_load_size}.ckpt')
        config.training_split = os.path.join(f'data/pctsp_custom/',f'{config.task}{config.task_size}_train_{config.data_dist}.txt')
        config.test_split = os.path.join(f'data/pctsp_custom/',f'{config.task}{config.task_size}_test_{config.data_dist}.txt')
    else:
        raise ValueError("wrong task comes")
    return config

def train_with_sweep(config=None):
    # basic Accelerate and logging setup
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    config = update_config(config, FLAGS)

    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    model, model_class, saving_mode = load_model(config)

    if config.kl_pretrain > 0 :
        pretrain_model, _, _ = load_model(config)
        pretrain_model.model.requires_grad_(False)
        
    if config.last_train or config.lora_rank > 0:
        model.model.requires_grad_(False)
    
    
    if config.sparse_factor>0 or config.task=='mis_sat' or config.task=='mis_er':
        sparse =True
    else :
        sparse =False
    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.inference_diffusion_steps * config.train.timestep_fraction)
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps, 
        kwargs_handlers=[DistributedDataParallelKwargs(
            find_unused_parameters=True,
            )]
    )
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    
    if config.last_train>0:
        model.to(accelerator.device, dtype=inference_dtype)

        for i in range(config.last_train):
            model.model.per_layer_out[-(i+1)].requires_grad_(True)   
            model.model.per_layer_out[-(i+1)].to(dtype=torch.float32)

            model.model.layers[-(i+1)].requires_grad_(True)
            model.model.layers[-(i+1)].to(dtype=torch.float32)

        model.model.out.requires_grad_(True)
        model.model.out.to(dtype=torch.float32)


    if config.lora_rank > 0:
        model = AddLora(model, config)

    ## Grdient stop for torch.compile

    if config.task=='tsp':
        model.model.layers[-1].U.requires_grad_(False)
        model.model.layers[-1].V.requires_grad_(False)
        model.model.layers[-1].norm_h.requires_grad_(False)

    if config.kl_aux>0 or config.use_critic > 0:
        model.model.generate_layer_aux(config.last_train)
        model.model.aux = True
    else:
        model.model.aux = False

    if config.kl_pretrain > 0 :
        pretrain_model.model.aux = model.model.aux

    aux_params, model_params = [], []
    for name, param in model.named_parameters():
        if 'aux' in name:
            aux_params.append(param)
            # print(f"Aux  - {name} (shape: {param.shape}, requires_grad: {param.requires_grad})")
        else:
            # print(f"Not aux  - {name} (shape: {param.shape}, requires_grad: {param.requires_grad})")
            model_params.append(param)
        
    if config.use_critic > 0:
        param_groups = [
        {'params': model_params, 'lr': config.learning_rate},
        {'params': aux_params, 'lr': config.learning_rate_aux}
        ]
    else:
        param_groups = [
        {'params': model.parameters(), 'lr': config.learning_rate}
        ]
    
    model.diffusion.Q_bar = model.diffusion.Q_bar.to(dtype=torch.float64)
    model.model = torch.compile(model.model)
    if config.kl_pretrain > 0 :
        pretrain_model.diffusion.Q_bar = pretrain_model.diffusion.Q_bar.to(dtype=torch.float64)
        pretrain_model.model = torch.compile(pretrain_model.model)

    #accelerator.device = device
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="CO_RLFINETUNE_LAST_DANCE_240924",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)


    # load scheduler, tokenizer and models.
    #model = model_class.load_from_checkpoint(ckpt_path, param_args=config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora model) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW


    optimizer = optimizer_cls(
        # model_params,
        param_groups, 
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    def print_optimizer_params(model, optimizer_1):
        print("Parameters managed by optimizer_1:")
        
        optimizer_param_ids = set(id(p) for group in optimizer_1.param_groups for p in group['params'])
        
        for name, param in model.named_parameters():
            if id(param) in optimizer_param_ids:
                print(f"  - {name} (shape: {param.shape}, requires_grad: {param.requires_grad})")
        
        total_params = sum(p.numel() for group in optimizer_1.param_groups for p in group['params'])
        print(f"\nTotal parameters managed by optimizer_1: {total_params}")
    

    autocast = accelerator.autocast if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    if config.use_env:
        if config.task=="tsp":
            train_dataloader = TSPGraphEnvironment(config.task_size,sparse_factor=config.sparse_factor)
        elif config.task=="mis_er":
            train_dataloader = MIS_ERGraphEnvironment()
    else:
        train_dataloader = model.train_dataloader()
    test_dataloader = model.test_dataloader()
    # rewards_mean, rewards_std = - model.test_dataset.cost_mean, model.test_dataset.cost_std
    model_args = copy.deepcopy(model.args)
    model_diffusion = model.diffusion
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)
    if config.kl_pretrain > 0 :
        pretrain_model = accelerator.prepare(pretrain_model)

    # top_checkpoints = []
    if config.task=="tsp":
        best_wo_2opt_score = np.inf
        best_w_2opt_score = np.inf
        best_reward_mean = - np.inf
    elif config.task=="mis_sat" or config.task=="mis_er":
        best_wo_2opt_score = -np.inf
        best_w_2opt_score = -np.inf
        best_reward_mean = - np.inf
        
    # Train!
    if config.sample.num_iters_per_epoch == -1: ## train_dataset for each epoch
        config.sample.num_iters_per_epoch = int(len(model.train_dataset)/config.batch_size)
    samples_per_epoch = (
        config.batch_size
        * accelerator.num_processes
        * config.sample.num_iters_per_epoch
    )

    total_train_batch_size = (
        config.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )


    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.batch_size}")
    logger.info(f"  Train batch size per device = {config.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.batch_size >= config.batch_size
    assert config.batch_size % config.batch_size == 0
    assert not (model_args.reward_baseline==2 and model_args.use_env)
    assert samples_per_epoch % total_train_batch_size == 0
    if (config.roll_out_num<=0 and config.reward_baseline==1):
        raise ValueError("roll_out_num should be larger than 0 when reward_baseline is 1", 'rolout', config.roll_out_num, 'baseline', config.reward_baseline )
    if config.reward_baseline==1 and config.sample.num_iters_per_epoch % config.roll_out_num!=0:
        raise ValueError("sample_iters should be multiple of roll_out_num when reward_baseline is 1")
        # assert (config.sample.num_iters_per_epoch % config.roll_out_num == 0)
    _ = None
    global_step = 0
    first_epoch = 0
    eval_count = 0
    time_eval = time.perf_counter()

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)

    if config.seed != -1:   
        set_seed(config.seed, device_specific=True)

    evaluate(model, model_diffusion, model_args, test_dataloader, _, accelerator,global_step, sparse, use_env=False, inference=True, reward_2opt=config.reward_2opt)
    train_actor = True
    print('train_actor',train_actor)
    first_epoch = 0
    save_freq = config.num_epochs // config.save_freq
    rewards_log = []
    rewards_std_log = []
    critic_loss_list = []
    reward_loss_list = []
    reward_std_avg = 0

    for epoch in range(first_epoch, config.num_epochs):
        # torch.distributed.barrier()

        if epoch != 0 and epoch % save_freq == 0:
            # and accelerator.is_main_process:
            accelerator.state.epoch = epoch
            accelerator.save_state()
        
        model.eval()
        with torch.no_grad():
            samples = []
                #################### SAMPLING ####################
            # prompts = []

            for i in tqdm(
                range(config.sample.num_iters_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                if config.reward_baseline!=1 or i % config.roll_out_num == 0:
                    if config.use_env:
                        batch = train_dataloader.get_batch(config.batch_size)
                    else:
                        batch = get_batch(train_dataloader)
                with autocast():
                    if model_args.task=='tsp':
                        latents, edge_index, log_probs, rewards, timesteps, reward_bonus, new_target, aux_pred  = difusco_with_logprob(
                            model,
                            model_diffusion,
                            model_args,
                            batch,
                            inference=False,
                            use_env = config.use_env,
                            sparse = sparse, 
                            reward_gap=config.reward_gap
                        )
                        points = batch[1]
                        if sparse :
                            points = batch[1].x.view([config.batch_size, -1, 2]).contiguous()
                            dist_mat = calculate_distance_matrix(batch[1].x.to('cpu'),edge_index.to('cpu'))
                            edge_index = torch.transpose(edge_index, 0, 1).contiguous()
                            edge_index = edge_index.view(config.batch_size, -1, 2) # [Batch, Num_instance*sparse_factor, 2]
                            edge_index = edge_index - config.task_size*torch.arange(config.batch_size).view([config.batch_size,1,1]).to(edge_index.device) 
                            latents = [latent.view(config.batch_size, -1) for latent in latents]
                        # print('ya')
                    elif model_args.task=='mis_sat' or model_args.task=='mis_er':
                        latents, edge_length, edge_index, log_probs, rewards, timesteps, new_target, aux_pred   = difusco_with_logprob_mis(model,
                            model_diffusion,
                            model_args,
                            batch,
                            inference=False,
                            sparse = sparse
                        )
                
                latents = torch.stack(
                    latents, dim=1
                )  # (batch_size, num_steps + 1, 4, 64, 64)
                log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
                timesteps = torch.stack(timesteps, dim=0).repeat(config.batch_size,1,1) # (batch_size, num_steps)  # (batch_size, num_steps)
                sample_dict={
                            "timesteps": timesteps.to('cpu'),
                            "latents": latents[
                                :, :-1
                            ].to('cpu'),  # each entry is the latent before timestep t
                            "next_latents": latents[
                                :, 1:
                            ].to('cpu'),  # each entry is the latent after timestep t
                            "log_probs": log_probs.to('cpu'),
                            "rewards": rewards.to('cpu'),
                                    }

                if config.kl_grdy>0 or config.kl_aux>0 or config.use_critic>0 or config.decoded_penalty>0:
                    if model_args.task=='tsp':
                        new_target = torch.stack(new_target, dim=0)
                        if not sparse :
                            new_target = new_target.unsqueeze(1).repeat(1, latents.shape[1]-1, 1, 1)
                        else :
                            new_target = new_target.unsqueeze(1).repeat(1, latents.shape[1]-1,1)
                    else:
                        new_target = new_target.unsqueeze(1).repeat(1, latents.shape[1]-1)
                    sample_dict['new_target'] = new_target.to('cpu')
                    
                    if config.kl_aux>0 or config.use_critic>0:
                        aux_pred = torch.stack(aux_pred, dim=1) 
                    
                    if config.use_critic>0:
                        if config.task=='tsp':
                            if sparse :
                                dist_unsqueeze = dist_mat.unsqueeze(1).repeat(1,model_args.inference_diffusion_steps)
                                aux_softmax = aux_pred.softmax(2)[:,:,1]
                                sample_dict['rewards_pred'] = (-torch.mul(dist_unsqueeze,aux_softmax)).reshape(model_args.batch_size,-1,model_args.inference_diffusion_steps).sum(1)
                            else:
                                dist_mat = calculate_distance_matrix(points.to('cpu'))
                                sample_dict['rewards_pred'] = -torch.mul(dist_mat.unsqueeze(1).repeat(1,model_args.inference_diffusion_steps,1,1).cpu(),aux_pred.softmax(2)[:,:,1,:,:]).sum(dim=[2,3])

                        else :
                            point_indicator = batch[2]
                            idx = 0
                            rewards_pred = []
                            for length in point_indicator:
                                rewards_pred.append(aux_pred.softmax(2)[:,:,1][idx:idx+length,:].sum(dim=0))
                                idx+=length
                            sample_dict['rewards_pred']=torch.stack(rewards_pred,dim=0)

                        if config.critic_tstart:
                            sample_dict['rewards_pred'][:,1:] = sample_dict['rewards_pred'][:,:1]
                            
                if model_args.reward_baseline==2 and not model_args.use_env:
                    sample_dict['opt_reward'] = - batch[-1].to('cpu')

                if config.task=="tsp":
                    sample_dict['points'] = points.to('cpu')
                    
                if model_args.reward_shaping:
                    reward_bonus = torch.stack(reward_bonus,dim=1)
                    sample_dict['rewards_bonus'] = reward_bonus.to('cpu')
                if sparse:
                    if config.task=='tsp':
                        sample_dict['edge_index'] = edge_index.to('cpu')
                # if sparse:
                #     edge_index = torch.transpose(edge_index, 0, 1).contiguous()
                    elif config.task=='mis_sat' or config.task=='mis_er':
                        point_indicator = batch[2]
                        idx_edge_index = 0
                        idx_point_indicator = 0
                        edge_index_list = []
                        for i in range(config.batch_size):
                            edge_index_list.append((edge_index[:,idx_edge_index:idx_edge_index+edge_length[i]]-idx_point_indicator).to('cpu'))
                            idx_point_indicator += point_indicator[i]
                            idx_edge_index += edge_length[i]
                        sample_dict['edge_index'] = edge_index_list

                        idx_latents = 0
                        latents_list = []
                        for i in range(config.batch_size):
                            latents_list.append(latents[idx_latents:idx_latents+point_indicator[i]][:,:-1].to('cpu'))
                            idx_latents += point_indicator[i]
                        sample_dict['latents'] = latents_list
                        sample_dict['point_indicator'] = point_indicator

                        idx_next_latents = 0
                        next_latents_list = []
                        for i in range(config.batch_size):
                            next_latents_list.append(latents[idx_next_latents:idx_next_latents+point_indicator[i]][:,1:].to('cpu'))
                            idx_next_latents += point_indicator[i]
                        sample_dict['next_latents'] = next_latents_list
                        
                        if model_args.kl_grdy>0 or model_args.kl_aux>0 or model_args.use_critic>0 or model_args.decoded_penalty>0:
                            idx_new_target = 0
                            new_target_list = []
                            for i in range(config.batch_size):
                                new_target_list.append(new_target[idx_new_target:idx_new_target+point_indicator[i]].to('cpu'))
                                idx_new_target += point_indicator[i]
                            sample_dict['new_target'] = new_target_list
                        
                samples.append(
                    sample_dict
                )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards = sample["rewards"]
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
            if model_args.use_critic :
                rewards_pred = sample["rewards_pred"]
                sample["rewards_pred"] = torch.as_tensor(rewards_pred, device=accelerator.device)
            
            
            if model_args.reward_baseline==2 and not model_args.use_env:
                opt_rewards = sample['opt_reward']
                sample['opt_reward'] = torch.as_tensor(opt_rewards, device=accelerator.device)
        
        # collate samples into dict where each entry has shape (num_batches_per_epoch * batch_size, ...)
        if config.task=="tsp":
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        elif config.task=='mis_sat' or config.task=='mis_er':
            samples_temp = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys() if k not in ["edge_index", "latents", "next_latents",'new_target']}

            samples_temp["edge_index"] = []
            samples_temp["latents"] = []
            samples_temp["next_latents"] = []
            if model_args.kl_grdy>0 or model_args.kl_aux>0 or model_args.use_critic>0 or model_args.decoded_penalty>0:
                samples_temp["new_target"] = []
            for s in samples:
                samples_temp["edge_index"].extend(s["edge_index"])
                samples_temp["latents"].extend(s["latents"])
                samples_temp["next_latents"].extend(s["next_latents"])
                if model_args.kl_grdy>0 or model_args.kl_aux>0 or model_args.use_critic>0 or model_args.decoded_penalty>0:
                    samples_temp["new_target"].extend(s["new_target"])
            
            samples = samples_temp

        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()
        
        rewards_log += rewards.tolist()
        
        if model_args.use_critic > 0:
            rewards_pred = accelerator.gather(samples["rewards_pred"]).cpu().numpy()
            rewards = np.repeat(np.expand_dims(rewards,axis=-1),repeats=model_args.inference_diffusion_steps,axis=1)
            
            rewards_adpated = rewards - rewards_pred
            if rewards_adpated.std() < 0.8 * rewards.std():
                rewards = rewards_adpated
        
        if model_args.reward_baseline==2 and not model_args.use_env:
            opt_rewards = accelerator.gather(samples['opt_reward']).cpu().numpy()

        if model_args.reward_shaping :
            rewards = (samples['rewards_bonus'] + torch.from_numpy(rewards).unsqueeze(1))

            samples["advantages"] = (rewards - rewards.mean()) / (rewards.std()+1e-6).to(accelerator.device)
            del samples["rewards"]
            del samples["rewards_bonus"]

        # advantages = (rewards - rewards_mean) / (rewards_std)
        else:
        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
            if model_args.reward_baseline==2 and not model_args.use_env:
                rewards = rewards - opt_rewards.reshape(rewards.shape)
            if model_args.reward_baseline==1:
                reward_shape = rewards.shape
                rewards = rewards.reshape(accelerator.num_processes, -1, model_args.roll_out_num, model_args.batch_size)
                rewards = rewards - rewards.mean(axis=-2, keepdims=True)
                rewards = rewards.reshape(reward_shape)
            
            advantages = (rewards - rewards.mean()) / (rewards.std()+1e-6)
            rewards_std_log.append(rewards.std() + 1e-6)
            
            if model_args.use_critic :
                samples["advantages"] = (
                    torch.as_tensor(advantages)
                    .reshape(accelerator.num_processes, -1 , model_args.inference_diffusion_steps)[accelerator.process_index]
                    .to(accelerator.device)
                )
            
            else:
                samples["advantages"] = (
                    torch.as_tensor(advantages)
                    .reshape(accelerator.num_processes, -1)[accelerator.process_index]
                    .to(accelerator.device)
                )
            
            del samples["rewards"]
            
        total_batch_size, num_timesteps, _ = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.batch_size * config.sample.num_iters_per_epoch
        )
        assert num_timesteps == config.inference_diffusion_steps

        #################### TRAINING ####################
        samples_cp = cp.deepcopy(samples)
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            
            perm = torch.randperm(total_batch_size, device='cpu')
            
            if config.task=="tsp":
                samples = {k: v[perm] for k, v in samples_cp.items()}
            elif config.task=="mis_sat" or config.task=="mis_er":
                samples = {k: sort_list_by_index(v, perm) if type(v)==type([]) else v[perm] for k, v in samples_cp.items()}

            # shuffle along time dimension independently for each sample
            if sparse:
                perms = torch.stack(
                    [
                        torch.arange(num_timesteps, device='cpu')
                        for _ in range(total_batch_size)
                    ]
                )
            else:
                perms = torch.stack(
                    [
                        torch.randperm(num_timesteps, device='cpu')
                        for _ in range(total_batch_size)
                    ]
                )
            # samples_cat_perm = dict()
            if not sparse:
                if model_args.reward_shaping :
                    for key in ["timesteps", "latents", "next_latents", "log_probs", "advantages",]:
                        samples[key] = samples[key][
                            torch.arange(total_batch_size, device='cpu')[:, None],
                            perms,
                        ]
                    if config.kl_grdy >0 or config.kl_aux>0 or config.use_critic>0 or config.decoded_penalty>0:
                        for key in ["new_target"]:
                            samples[key] = samples[key][
                                torch.arange(total_batch_size, device='cpu')[:, None],
                                perms,
                            ]
                else :
                    for key in ["timesteps", "latents", "next_latents", "log_probs",]:
                        samples[key] = samples[key][
                            torch.arange(total_batch_size, device='cpu')[:, None],
                            perms,
                        ]
                    if config.kl_grdy >0 or config.kl_aux>0 or config.use_critic>0 or config.decoded_penalty>0:
                        if config.use_critic>0:
                            key_list = ["new_target", "advantages"]
                        else:
                            key_list = ["new_target"]
                        for key in key_list:
                            samples[key] = samples[key][
                                torch.arange(total_batch_size, device='cpu')[:, None],
                                perms,
                            ]
            # rebatch for training
            if config.task=="tsp":
                samples_batched = {
                    k: v.reshape(-1, config.batch_size, *v.shape[1:])
                    for k, v in samples.items()
                }
            elif config.task=='mis_sat' or config.task=='mis_er':
                samples_batched = {
                    k: v.reshape(-1, config.batch_size, *v.shape[1:]) if not isinstance(v, list) else reshape_list(v, batch_size=config.batch_size)
                    for k, v in samples.items()
                }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if sparse:
                    if config.task=='tsp':
                        sample['edge_index'] += config.task_size*torch.arange(config.batch_size).view([config.batch_size,1,1]).to(sample['edge_index'].device)
                        sample['edge_index'] = sample['edge_index'].view(-1, 2).transpose(0,1).contiguous()
                        sample['points'] = sample['points'].view(-1, 2).contiguous()
                        sample['latents'] = sample['latents'].transpose(1,2).reshape(-1, config.inference_diffusion_steps).contiguous()
                        sample['next_latents'] = sample['next_latents'].transpose(1,2).reshape(-1, config.inference_diffusion_steps).contiguous()

                        if config.kl_grdy>0 or config.kl_aux>0 or config.use_critic>0 or config.decoded_penalty>0:
                            sample['new_target'] = sample['new_target'].transpose(1,2).reshape(-1, config.inference_diffusion_steps).contiguous()
                    elif config.task=='mis_sat' or config.task=='mis_er':
                        edge_index_shift = 0
                        edge_index_list = []
                        for edge_index_i in sample['edge_index']:
                            index_shift = edge_index_i.max()+1
                            edge_index_i = cp.deepcopy(edge_index_i) + edge_index_shift
                            edge_index_list.append(edge_index_i)
                            edge_index_shift += index_shift

                        sample['edge_index'] = torch.cat(edge_index_list, dim=1).contiguous()
                        sample['latents'] = torch.cat(sample['latents'], dim=0).contiguous()
                        sample['next_latents'] = torch.cat(sample['next_latents'], dim=0).contiguous()
                        if config.kl_grdy>0 or config.kl_aux>0 or config.use_critic>0 or config.decoded_penalty>0:
                            sample['new_target'] = torch.cat(sample['new_target'], dim=0).contiguous()
                sample = {k: v.to(accelerator.device) for k, v in sample.items()}
                
                if hasattr(train_dataloader, 'end_of_dataloader') and train_dataloader.end_of_dataloader:
                    train_dataloader.end_of_dataloader = False
                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(model):
                        with autocast():
                            if sparse:
                                t_start, t_target =  torch.tensor([sample['timesteps'][0,j,0]]), torch.tensor([sample['timesteps'][0,j,1]])
                            else:
                                t_start, t_target = sample['timesteps'][:,j,0], sample['timesteps'][:,j,1]
                            
                            # print('config.task', config.task)
                            if config.task=='tsp':
                                _, log_prob, _, x0_pred, x0_pred_aux = categorical_denoise_step(
                                    model,
                                    model_diffusion,
                                    model_args,
                                    sample['points'], 
                                    sample["latents"][:, j], 
                                    t_start, 
                                    model.device, 
                                    edge_index= sample['edge_index'] if sparse else None, 
                                    target_t=t_target,
                                    next_xt= sample["next_latents"][:, j],
                                    batch_t=not sparse,
                                    inference=False,
                                    sparse=sparse,
                                    aux=True
                                    )

                            elif config.task =='mis_er' or config.task=='mis_sat':
                                if config.diffusion_type=='categorical':
                                    _, log_prob, _,x0_pred, x0_pred_aux = categorical_denoise_step_mis(
                                    model,
                                    model_diffusion,
                                    model_args,
                                    sample["latents"][:, j], 
                                    t_start, 
                                    model.device, 
                                    edge_index= sample['edge_index'] if sparse else None, 
                                    target_t=t_target,
                                    next_xt= sample["next_latents"][:, j],
                                    batch_t=not sparse,
                                    inference=False,
                                    sparse=sparse, 
                                    point_indicator = sample['point_indicator'],
                                    aux=True
                                    )
                                elif config.diffusion_type=='gaussian':
                                    _, log_prob, _ = gaussian_denoise_step_mis(
                                    model,
                                    model_diffusion,
                                    model_args,
                                    sample["latents"][:, j], 
                                    t_start, 
                                    model.device, 
                                    edge_index= sample['edge_index'] if sparse else None, 
                                    target_t=t_target,
                                    next_xt= sample["next_latents"][:, j],
                                    batch_t=not sparse,
                                    inference=False,
                                    sparse=sparse, 
                                    point_indicator = sample['point_indicator']
                                    )
                            else:
                                raise ValueError("task should be tsp or mis_er or mis_sat")

                        critic_loss = 0
                        if model_args.use_critic:
                            if model_args.task == 'tsp' :
                                if model_args.sparse_factor>0 :
                                    dist_mat = calculate_distance_matrix(sample['points'].to(accelerator.device), sample['edge_index'])
                                    reward_pred = torch.mul(dist_mat, x0_pred_aux.softmax(1)[:,1]).view([config.batch_size, -1]).sum(-1)
                                    reward_decoded = torch.mul(dist_mat, sample['new_target'][:,j].long()).view([config.batch_size, -1]).sum(-1)
                                    loss_mse = nn.MSELoss()
                                    reward_loss = np.sqrt(loss_mse(reward_pred.detach(), reward_decoded.detach()).item())
                                    if model_args.use_critic==1:
                                        if config.use_weighted_critic:
                                            loss_func_critic = nn.CrossEntropyLoss(reduction='none')
                                            critic_loss_raw = loss_func_critic(x0_pred_aux, sample['new_target'][:, j].long())
                                            critic_loss = (critic_loss_raw * 2*dist_mat).mean()
                                            
                                        else:
                                            loss_func = nn.CrossEntropyLoss()
                                            critic_loss = loss_func(x0_pred_aux, sample['new_target'][:,j].long())
                                    #reward_loss = 0
                                    elif model_args.use_critic==2:
                                        critic_loss = loss_mse(reward_pred, reward_decoded)
                                        
                                else:
                                    dist_mat = calculate_distance_matrix(sample['points'].to(accelerator.device)) # [B,N,N] <- [B,N,2]
                                    reward_pred = torch.mul(dist_mat, x0_pred_aux.softmax(1)[:,1]).sum(dim=[1,2]) # ([B,N,N] * [B,N,N]).sum(1,2)
                                    reward_decoded = torch.mul(dist_mat, sample['new_target'][:,j].long()).sum(dim=[1,2])
                                    # print('reward_decoded', reward_decoded)
                                    loss_mse = nn.MSELoss()
                                    reward_loss = np.sqrt(loss_mse(reward_pred.detach(), reward_decoded.detach()).item())
                                    if model_args.use_critic==1:
                                        if config.use_weighted_critic:
                                            loss_func_critic = nn.CrossEntropyLoss(reduction='none')
                                            critic_loss_raw = loss_func_critic(x0_pred_aux, sample['new_target'][:, j].long())
                                            critic_loss = (critic_loss_raw * 2*dist_mat).mean()
                                        else:
                                            loss_func = nn.CrossEntropyLoss()
                                            critic_loss = loss_func(x0_pred_aux, sample['new_target'][:,j].long())
                                    elif model_args.use_critic==2:
                                        critic_loss = loss_mse(reward_pred, reward_decoded)
                            #MIS case
                            else:
                                idx = 0
                                reward_pred = []
                                reward_decoded = []
                                for length in sample['point_indicator']:
                                    reward_pred.append(x0_pred_aux.softmax(1)[:,1][idx:idx+length].sum(dim=0))
                                    reward_decoded.append(sample['new_target'][:,j][idx:idx+length].sum(dim=0))
                                    idx+=length
                                reward_pred=torch.stack(reward_pred,dim=0)
                                reward_decoded = torch.stack(reward_decoded,dim=0)
                                loss_mse = nn.MSELoss()
                                reward_loss = np.sqrt(loss_mse(reward_pred.detach(), reward_decoded.detach()).item())
                                if model_args.use_critic==1:
                                    if config.use_weighted_critic:
                                        loss_func_critic = nn.CrossEntropyLoss(reduction='none')
                                        critic_loss_raw = loss_func_critic(x0_pred_aux, sample['new_target'][:, j].long())
                                        critic_loss = (critic_loss_raw).mean()
                                    else:
                                        loss_func = nn.CrossEntropyLoss()
                                        critic_loss = loss_func(x0_pred_aux, sample['new_target'][:,j].long())
                                elif model_args.use_critic==2:
                                    critic_loss = loss_mse(reward_pred, reward_decoded)
                            
                            if model_args.critic_tstart:
                                if t_start == 1000:
                                    alpha = 1.0
                                else:
                                    alpha = 0
                                critic_loss = alpha * critic_loss
                                   
                        if train_actor:
                            if model_args.reward_shaping or model_args.use_critic:
                                advantages = torch.clamp(
                                    sample["advantages"][:,j],
                                    -config.train.adv_clip_max,
                                    config.train.adv_clip_max,
                                )
                            else :
                                advantages = torch.clamp(
                                    sample["advantages"],
                                    -config.train.adv_clip_max,
                                    config.train.adv_clip_max,
                                )
                            
                            ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                            unclipped_loss = -advantages * ratio
                            clipped_loss = -advantages * torch.clamp(
                                ratio,
                                1.0 - config.train.clip_range,
                                1.0 + config.train.clip_range,
                            )
                            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            
                            if model_args.kl_grdy >0 :
                                loss_func = nn.CrossEntropyLoss(reduction='none')

                                kl_grdy_loss = loss_func(x0_pred, sample['new_target'][:,j].long())
                                kl_grdy_loss = kl_grdy_loss.mean(tuple(range(1, kl_grdy_loss.ndim)))
                                loss +=  - (advantages * model_args.kl_grdy*kl_grdy_loss).mean()
                                
                            if model_args.kl_pretrain >0 :
                                if config.task =='mis_er' or config.task=='mis_sat':
                                    x0_pred_pretrain = pretrain_model.forward(
                                                sample["latents"][:, j].to(model.device, pretrain_model.dtype), 
                                                t_start.to(model.device), 
                                                edge_index= sample['edge_index'] if sparse else None, )
                                else :
                                    x0_pred_pretrain = pretrain_model.forward(
                                            sample['points'].float().to(model.device),
                                            (sample["latents"][:, j]).float().to(model.device),
                                            t_start.float().to(model.device),
                                            edge_index.long().to(model.device) if edge_index is not None else None,)
                                loss_func = nn.CrossEntropyLoss()
                                kl_pretrain_loss = loss_func(x0_pred,F.softmax(x0_pred_pretrain[0],dim=1))
                                loss += model_args.kl_pretrain*kl_pretrain_loss
                            
                            
                            if model_args.kl_aux>0:
                                loss_func = nn.CrossEntropyLoss()
                                aux_loss = loss_func(x0_pred_aux, sample['new_target'][:,j].long())
                                loss += model_args.kl_aux*aux_loss


                            # debugging values
                            # John Schulman says that (ratio - 1) - log(ratio) is a better
                            # estimator, but most existing code uses this so...
                            # http://joschu.net/blog/kl-approx.html
                            info["approx_kl"].append(
                                0.5
                                * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                            )
                            info["clipfrac"].append(
                                torch.mean(
                                    (
                                        torch.abs(ratio - 1.0) > config.train.clip_range
                                    ).float()
                                )
                            )
                            info["loss"].append(loss)

                        else:
                            loss = 0


                        if train_actor:
                            loss = loss + config.critic_ratio*critic_loss
                            accelerator.backward(loss)
                        else:
                            accelerator.backward(critic_loss)

                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                model.parameters(), config.train.max_grad_norm
                            )
                            accelerator.wait_for_everyone()
                        

                        optimizer.step()
                        optimizer.zero_grad()
                        
                        if model_args.use_critic > 0:
                            critic_loss_list.append(critic_loss.detach().item())
                            reward_loss_list.append(reward_loss)


                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)


            assert accelerator.sync_gradients
            
        if config.eval_step < 0:
            eval_count += 1

            log_info = dict()
            if config.use_critic:
                log_info['critic_loss'] = np.array(critic_loss_list).reshape([-1, num_train_timesteps]).mean()
                log_info['reward_loss'] = np.array(reward_loss_list).reshape([-1, num_train_timesteps]).mean()
                accelerator.log(log_info, step=global_step)
            if not train_actor:
                print('critic_loss_list', np.array(critic_loss_list).reshape([-1, num_train_timesteps]).mean(), 'reward_loss_list', np.array(reward_loss_list).reshape([-1, num_train_timesteps]).mean())
            critic_loss_list = []
            reward_loss_list = []

            if eval_count % - config.eval_step == 0:
                
                _ = None
                time_eval = time.perf_counter()
                wo_2opt_cost, w_2opt_cost = evaluate(model,model_diffusion,model_args,test_dataloader, _, accelerator,global_step, sparse, use_env=False, inference=True, reward_2opt=config.reward_2opt)

                reward_mean = np.mean(rewards_log)
                reward_std = np.mean(rewards_std_log)
                rewards_log = []
                rewards_std_log = []
                if accelerator.is_main_process:
                    print('epoch', epoch, "evaluation done", time.perf_counter()-time_eval, 'reward_mean', reward_mean, 'reward_std', reward_std)
                log_info = {"reward_mean": reward_mean, "epcoh": epoch, "reward_std": reward_std,}
                accelerator.log(log_info, step=global_step)

                check_save = False
                if best_reward_mean < reward_mean:
                    if accelerator.is_main_process:
                        print('the best model score updated and saved', best_reward_mean, '-->', reward_mean)
                    best_reward_mean = reward_mean
                    check_save = True

                if config.task=='tsp':
                    if best_wo_2opt_score>wo_2opt_cost or best_w_2opt_score>w_2opt_cost:
                        if accelerator.is_main_process:
                            print('the best model score updated and saved', best_wo_2opt_score, '-->',  min(wo_2opt_cost, best_wo_2opt_score))
                            print('the best model score updated and saved', best_w_2opt_score, '-->',  min(w_2opt_cost, best_w_2opt_score))

                        best_wo_2opt_score = min(wo_2opt_cost, best_wo_2opt_score)
                        best_w_2opt_score = min(w_2opt_cost, best_w_2opt_score)
                        check_save = True

                elif config.task=='mis_sat' or config.task=='mis_er':
                    if best_wo_2opt_score<wo_2opt_cost:
                        
                        if accelerator.is_main_process:
                            print('the best model score updated and saved', best_wo_2opt_score, '-->',  max(wo_2opt_cost, best_wo_2opt_score))
                        best_wo_2opt_score = max(wo_2opt_cost, best_wo_2opt_score)
                        check_save = True

                if check_save:
                    accelerator.state.epoch = epoch
                    accelerator.save_state()

                if accelerator.is_main_process:
                    print("evaluation done", time.perf_counter()-time_eval)
    wandb.finish()
import gc

def print_gpu_memory_usage(string, model=None):
    print("GPU Memory Usage:" + string)
    if not model==None:
        total_memory = 0
        
        for name, param in model.named_parameters():
            if param.is_cuda:
                total_memory += param.element_size() * param.nelement()
        
        print(f" Total GPU Memory Usage: {total_memory / 1024 / 1024:.2f} MB")
        
        memory_usage = []
        for name, param in model.named_parameters():
            if param.is_cuda:
                memory = param.element_size() * param.nelement()
                memory_usage.append((name, param.size(), memory))
        
        memory_usage.sort(key=lambda x: x[2], reverse=True)
        
        for name, size, memory in memory_usage:
            print(f" {name} ({size}) - {memory / 1024 / 1024:.2f} MB")
    else:
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f" {type(obj).__name__} ({obj.size()}) - {obj.element_size() * obj.nelement() / 1024 / 1024:.2f} MB")




def sort_list_by_index(lst, index_list):
    return [lst[i] for i in index_list]
    
def reshape_list(list_to_reshape, batch_size):
    """
    change initial shape of list
    """
    if len(list_to_reshape)%batch_size!=0:
        raise ValueError("batch_size should be divisible by the length of list")
    return [list_to_reshape[i*batch_size:(i+1)*batch_size] for i in range(len(list_to_reshape)//batch_size)]

def evaluate(model,model_diffusion,model_args,test_dataloader, _,accelerator,global_step, sparse, use_env, inference, reward_2opt):
    
    all_sl_loss = []
    model.eval()
    num_batch = 0
    with torch.no_grad():
        time1 = time.perf_counter()
        all_gt_costs,all_wo_2opt_costs,all_solution_costs = [],[],[]
        if model_args.task=='tsp':
            for batch in test_dataloader:
                num_batch += 1
                gt_costs,wo_2opt_costs,solution_costs = difusco_with_logprob(model,model_diffusion,model_args,batch,inference,use_env,sparse, reward_2opt=True)
                all_gt_costs += gt_costs
                all_wo_2opt_costs += wo_2opt_costs
                all_solution_costs += solution_costs
                if model_args.print_sl_loss:
                    sl_loss = model.categorical_training_step(batch, 0).detach()
                    # sl_loss = model.module.categorical_training_step(batch, 0).detach()
                    all_sl_loss += [sl_loss]
        elif model_args.task=='mis_sat' or model_args.task=='mis_er':
            for batch in test_dataloader:
                gt_costs, solved_costs_xt, solved_costs_prob = difusco_with_logprob_mis(model, model_diffusion, model_args, batch, inference, sparse, decode_heatmap=model_args.reward_2opt)

                all_wo_2opt_costs += solved_costs_xt
                all_solution_costs += solved_costs_prob
    all_gt_costs = accelerator.gather(torch.as_tensor(all_gt_costs, device=accelerator.device)).cpu().numpy()
    all_wo_2opt_costs = accelerator.gather(torch.as_tensor(all_wo_2opt_costs, device=accelerator.device)).cpu().numpy()
    all_solution_costs = accelerator.gather(torch.as_tensor(all_solution_costs, device=accelerator.device)).cpu().numpy()
    if model_args.print_sl_loss:
        all_sl_loss = accelerator.gather(torch.as_tensor(all_sl_loss, device=accelerator.device)).cpu().numpy()
        # if not print_log:
    log_cost = {
            "num of samples": len(all_solution_costs),
            "test_gt_costs": np.mean(all_gt_costs),
            "test_model_costs": np.mean(all_solution_costs),
            "test_model_costs_wo2opt": np.mean(all_wo_2opt_costs),
        }
    if model_args.print_sl_loss:
        log_cost['test_supervised_loss'] = torch.tensor(all_sl_loss).mean()

    accelerator.log(
            log_cost,
            step=global_step,
        )
    
    # if reward_2opt:
    return np.mean(all_wo_2opt_costs), np.mean(all_solution_costs)

            
def load_model(config):
    """
    Load model and model calss
    """
    from difusco.pl_tsp_model import TSPModel
    from difusco.pl_mis_model import MISModel
    # from difusco.pl_tsp_model_free import TSPModelFreeGuide

    # if config.parallel_sampling!=1:
    #     raise Exception("parallel_sampling is removed!")
    if config.task == "tsp":
        if config.diffusion_type == "gaussian" or config.diffusion_type == "categorical":
            model_class = TSPModel
            saving_mode = "min"
        elif config.diffusion_type == "classifier":
            raise NotImplementedError
        elif config.diffusion_type == "reward":
            raise NotImplementedError
        else:
            raise NotImplementedError
        
    elif config.task == "mis_sat" or config.task == "mis_er":
        model_class = MISModel
        saving_mode = "max"
    
    elif config.task == "pctsp":
        raise NotImplementedError


    model = model_class.load_from_checkpoint(config.ckpt_path, param_args=config)
    #model = model_class(param_args=config)

    return model, model_class, saving_mode

def print_all_submodules(model):
    print("모델의 모든 서브 모듈:")
    for name, module in model.named_modules():
        print(f"{name}: {module}")

def check_gradient(model):
    """
    Check gradient
    """
    params_with_grad = []
    params_without_grad = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            params_with_grad.append(name)
        else:
            params_without_grad.append(name)
    return params_with_grad, params_without_grad

def compute_torch_C(torch_A, torch_B):
    torch_C = torch.zeros_like(torch_B)
    
    indices = torch.nonzero(torch_B, as_tuple=True)
    
    if len(indices[0]) == 0:
        return torch_C
    
    batch_indices = indices[0]
    j_indices = indices[1]
    k_indices = indices[2]
    
    condition = torch_A[batch_indices, j_indices, k_indices] >= torch_A[batch_indices, k_indices, j_indices]
    
    valid_indices = [batch_indices[condition], j_indices[condition], k_indices[condition]]
    
    torch_C[valid_indices[0], valid_indices[1], valid_indices[2]] = torch_B[valid_indices[0], valid_indices[1], valid_indices[2]]
    
    return torch_C

def compute_torch_C_dense(torch_A, torch_B):
    mask = torch_A >= torch_A.transpose(-2, -1)
    torch_C = torch_B * mask
    return torch_C

def main(_):
    if FLAGS.use_sweep:
        sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'reward_mean',
                'goal': 'maximize'
            },
            'parameters': {
                'train.learning_rate': {
                    'values': [3e-7, 1e-6, 3e-6]
                },
                
            }
        }
        print('sweep mode', sweep_config)

        sweep_id = wandb.sweep(sweep_config, project='ddpo-pytorch')

        wandb.agent(sweep_id, function=train_with_sweep)
    else:
        train_with_sweep()


if __name__ == "__main__":
    app.run(main)
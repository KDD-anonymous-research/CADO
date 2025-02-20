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

from difusco.utils.eval_utils import evaluate_test, setup_logging, format_log_message
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
import copy



import os
time_init = time.perf_counter()

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base_co.py", "Training configuration.")

flags.DEFINE_bool('use_sweep', False, 'Whether to use Weights & Biases sweep for hyperparameter search.')
flags.DEFINE_string('run_name', "", 'Your name')
flags.DEFINE_string('task', "tsp", 'tsp, mis_sat, mis_er')
flags.DEFINE_integer('task_sweep', 0, 'sweep to mis_sat vs mis_er')
flags.DEFINE_integer('task_size', 50, "task size to solve")
flags.DEFINE_integer('task_load_size', 50, "the task size that should be loaded ckpt")

flags.DEFINE_integer('batch_size', 1, "a") # Evaluation이나 처음 sampling을 할 때 사용하는 세팅
flags.DEFINE_integer('gradient_accumulation_steps',1,"a") #100의 약수
flags.DEFINE_integer('sample_iters', 2,"a")
# flags.DEFINE_integer('onpolicy', 0, "0: Importance sampling, 1: onpolicy")
flags.DEFINE_integer('train_iters', 1, "a")
flags.DEFINE_integer('num_epochs', 4,"a")
flags.DEFINE_integer('print_sl_loss', 0,"a")

flags.DEFINE_integer('eval_step', -5,"number of steps to evaluate, -1: evaluation for each epoch, 1000: evaluateion whenever for 1000 global steps")

flags.DEFINE_integer('num_workers', 1,"a")
flags.DEFINE_integer('inference_diffusion_steps', 20, "a")

flags.DEFINE_string('tsp_decoder', 'cython_merge', "cython_merge or am_decoding or farthest")

flags.DEFINE_integer('critic_tstart', 0, "0: baseline for divided by all time step, 1: basline for not all time ")
flags.DEFINE_integer('sparse_factor', -1, "a")
flags.DEFINE_integer('two_opt_iterations', 5000, "a")
flags.DEFINE_integer('reward_2opt', 0, "1: with 2opt, 0: without 2opt")
flags.DEFINE_bool('reward_gap', False, "False: do not use reward gap, 1: use reward gap")   
flags.DEFINE_integer('reward_baseline', 0, "0: normal reward, 1: reward with self baseline, 2: reward with optimal baseline")
flags.DEFINE_integer('roll_out_num', 32, "Num of roll outs for reward baseline")

flags.DEFINE_string('subopt', "", "name for reading")
flags.DEFINE_bool("use_activation_checkpoint", False, "use activation checkpoint")


flags.DEFINE_string('resume_from', "", 'resume from checkpoint')

# flags.DEFINE_bool("lora", True, "use lora")
flags.DEFINE_string("mixed_precision", 'no', "check mixed precision, no or fp16")
flags.DEFINE_float('learning_rate', 1e-5, "learning rate")
flags.DEFINE_float('learning_rate_aux', 2e-4, "learning rate")
flags.DEFINE_bool('use_env', False, 'generate datasamples during training')
# flags.DEFINE_bool('use_env', False, 'generate datasamples during training')

flags.DEFINE_integer('last_train', 1, 'train only last layer')
flags.DEFINE_integer("lora_rank", 2, "use lora")
flags.DEFINE_integer("lora_new", 1, "0: lora old versoin, 1: lora new version")

flags.DEFINE_integer("lora_range", 1, "use lora")
flags.DEFINE_integer('num_testset', -1, 'number of testset')
flags.DEFINE_integer('num_trainset', -1, 'number of trainset')

flags.DEFINE_integer("use_critic",0, "use critic: 0: REINFORCE, 1: critic for decoded sol, 2: value only")
flags.DEFINE_integer("use_weighted_critic", 0, "0: just kl, 1: weighted critic")


flags.DEFINE_integer('seed', -1, 'seed for random number generator, -1: random seed')
flags.DEFINE_bool('reward_shaping', False, 'use reward shaping')
flags.DEFINE_float('kl_grdy', 0.00, 'kl divergence between greedy decoded output')
flags.DEFINE_float('kl_pretrain', 0.00, 'kl divergence between original difusco output')
flags.DEFINE_float('kl_aux', 0.00, 'kl divergence between auxiliary')
flags.DEFINE_float('critic_ratio', 1., 'critic learning rate ratio compared to actor')

flags.DEFINE_integer("rewrite_steps", 0, "rewrite")
flags.DEFINE_integer("rewrite", 1, "rewrite")
flags.DEFINE_float("rewrite_ratio", 0.25, "save frequency")
flags.DEFINE_integer("inference_steps", 10, "inference steps rewrite")
flags.DEFINE_integer("parallel_sampling", 3,"parallel sampling")
flags.DEFINE_integer("tsplib", 0, "0: not use tsplib, 1: use tsplib")
flags.DEFINE_string("name_exper", '',"name of experiment")

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
    config.rewrite_steps = input_var.rewrite_steps
    config.rewrite = input_var.rewrite
    config.rewrite_ratio = input_var.rewrite_ratio
    config.inference_steps = input_var.inference_steps
    config.parallel_sampling = input_var.parallel_sampling
    config.use_activation_checkpoint = input_var.use_activation_checkpoint
    config.kl_pretrain = input_var.kl_pretrain
    config.kl_grdy = input_var.kl_grdy
    config.kl_aux = input_var.kl_aux
    config.reward_baseline = input_var.reward_baseline
    config.roll_out_num = input_var.roll_out_num
    config.use_critic = input_var.use_critic
    config.critic_ratio = input_var.critic_ratio
    config.use_weighted_critic = input_var.use_weighted_critic
    config.critic_tstart = input_var.critic_tstart
    config.task_sweep = input_var.task_sweep
    config.lora_new = input_var.lora_new
    config.tsplib = input_var.tsplib

    config.run_name = f'{input_var.run_name}{input_var.task}{input_var.task_size}_steps{input_var.inference_diffusion_steps}_bs{config.batch_size},st{input_var.sample_iters},ti{input_var.train_iters},accu{config.train.gradient_accumulation_steps},lr{input_var.learning_rate}'
    
    num_processes = max(torch.cuda.device_count(), 1)

    config.run_name += f'_tb{num_processes*config.batch_size*config.train.gradient_accumulation_steps}'

    if config.last_train>0 :
        config.run_name += f"_last{config.last_train}"
    if config.lora_rank > 0:
        config.run_name += f"_lora{config.lora_rank}"
    if config.lora_rank <= 0 and config.last_train <= 0:
        config.run_name += "_full"
    if config.kl_grdy > 0:
        config.run_name += f"_klg{config.kl_grdy}"
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
        if config.tsplib:
            if config.task_size==100:
                config.test_split = os.path.join(f'data/tsp_custom/', 'tsplib50-200.txt')
            elif config.task_size==500:
                config.test_split = os.path.join(f'data/tsp_custom/', 'tsplib200-1000.txt')
            else:
                raise ValueError("wrong task size comes")
        else:
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

    log_filename = config.task + str(config.task_size) + config.tsp_decoder + '_step' + str(config.inference_diffusion_steps) + '_seq' + str(config.sequential_sampling) + '_pal' + str(config.parallel_sampling) +'_2opt'+ str(config.reward_2opt) + '_rew' + str(config.rewrite) + '_ratio' + str(config.rewrite_ratio) +'_'+ unique_id + "log.txt"
    log_filename = os.path.join('/lab-di/squads/diff_nco/eval_logs/', log_filename)
    logger_txt = setup_logging(log_filename)

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



    # if config.use_critic > 0 :
    #     critic_params = [p for n, p in model.named_parameters() if 'aux' in n]
    #     critic_optimizer = optimizer_cls(
    #         critic_params,
    #         lr=config.learning_rate,
    #         betas=(config.train.adam_beta1, config.train.adam_beta2),
    #         weight_decay=config.train.adam_weight_decay,
    #         eps=config.train.adam_epsilon,
    #     )
    #     model_params = [p for n, p in model.named_parameters() if 'aux' not in n]
    # else:
    #     model_params = model.parameters()

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
    
    # print_optimizer_params(model, optimizer)
    # if config.use_critic > 0 :
    #     print_optimizer_params(model, critic_optimizer)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = accelerator.autocast if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    if config.use_env:
        train_dataloader = TSPGraphEnvironment(config.task_size,sparse_factor=config.sparse_factor)
    else:
        train_dataloader = model.train_dataloader()
    test_dataloader = model.test_dataloader()
    model_args = copy.deepcopy(model.args)
    model_diffusion = model.diffusion
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    
    if config.kl_pretrain > 0 :
        pretrain_model = accelerator.prepare(pretrain_model)
    # if config.use_critic > 0 :
    #     critic_optimizer = accelerator.prepare(critic_optimizer)
        
        
    # accelerator.load_state('~/pcb/rl_finetuning/ddpo_co/logs/tsp100_steps7_bs4,st3,ti2_lr0.001_useenv_last_2024.05.08_13.56/checkpoints/checkpoint_1')

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
    time2 = time.perf_counter()
    print("Initialization requires", time2-time_init)
    test_scores = []
    time_list = []
    log_filename = config.task + str(config.task_size) + '_' + config.tsp_decoder + '_denoise' + str(config.inference_diffusion_steps) + '_seq' + str(config.sequential_sampling) + '_pal' + str(config.parallel_sampling) + '_rewrite' + str(config.rewrite) + '_rewrite_ratio' + str(config.rewrite_ratio) + '_rwrite_steps' + str(config.rewrite_steps) + "log.txt"
    log_filename = os.path.join('/lab-di/squads/diff_nco/eval_logs/', log_filename)
    
    if config.seed != -1:   
        print('seed', config.seed)
        set_seed(config.seed, device_specific=True)

    for _ in range(4):
        time_eval_start = time.perf_counter()
        score = evaluate_test(model, model_diffusion, model_args, test_dataloader, _, accelerator,global_step, sparse, use_env=False, inference=True, reward_2opt=config.reward_2opt)
        test_scores.append(score)
        time_list.append(time.perf_counter()-time_eval_start)
    
    test_scores = accelerator.gather(torch.tensor(test_scores).to(accelerator.device)).cpu()

    log_message = format_log_message('test_scores', test_scores.mean(), 'time', np.mean(time_list[1:]), test_scores,   'time_list',time_list)
    if accelerator.is_main_process:
        logger_txt.info(log_message)
        # logger_txt.info(format_log_message(config))
    exit(0)


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
                    # sl_loss = model.categorical_training_step(batch, 0).detach()
                    sl_loss = model.module.categorical_training_step(batch, 0).detach()
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

    # if config.parallel_sampling!=1:
    #     raise Exception("parallel_sampling is removed!")
    if config.task == "tsp":
        if config.diffusion_type == "gaussian" or config.diffusion_type == "categorical":
            if config.return_condition :
                raise NotImplementedError
            else :
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
        if config.diffusion_type == "gaussian" or config.diffusion_type == "categorical":
            if config.return_condition :
                raise NotImplementedError
            else :
                raise NotImplementedError
            saving_mode = "min"

        else:
            raise NotImplementedError

    else: 
        raise NotImplementedError

    model = model_class.load_from_checkpoint(config.ckpt_path, param_args=config)

    return model, model_class, saving_mode

def print_all_submodules(model):
    print("Print all submodules:")
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
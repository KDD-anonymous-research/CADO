# from argparse import ArgumentParser
# cado_developer1 = True

# def arg_parser():
#     parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')
#     parser.add_argument('--task', type=str, default="tsp", help='tsp, mis, pctsp')
#     if cado_developer1:
#         parser.add_argument('--storage_path', type=str, default="/lab-di/squads/diff_nco")
#     else:
#         parser.add_argument('--storage_path', type=str, default="/workspace/pair/squads/diff_nco")
#     parser.add_argument('--log_dir', type=str, default="logs_cado_developer1/logs_240226_pctsp") # logs_cado_developer1/logs_0111
# #     parser.add_argument('--log_dir', type=str, default="logs_cado_developer2")
#     parser.add_argument('--data_dist', type=str, default='optimal',help="optimal, fullrand, halfrand, subtourrand")
#     parser.add_argument('--task_size', type=str, default="100")
#     parser.add_argument('--validation_examples', type=int, default=-1, help='-1: all data, k: use first k samples for validation')
#     parser.add_argument('--fp16', action='store_true')
#     parser.add_argument('--use_activation_checkpoint', action='store_true')
#     parser.add_argument('--batch_size', type=int, default=2)
#     parser.add_argument('--num_workers', type=int, default=1)

    
    
#     parser.add_argument('--diffusion_type', type=str, default='categorical')
#     parser.add_argument("--diffusion_space", type=str, default="discrete")
#     parser.add_argument('--diffusion_schedule', type=str, default='linear')
#     parser.add_argument('--diffusion_steps', type=int, default=1000)
#     parser.add_argument('--inference_diffusion_steps', type=int, default=5)
#     parser.add_argument('--inference_schedule', type=str, default='cosine')
#     parser.add_argument('--inference_trick', type=str, default="ddim")
#     parser.add_argument('--sequential_sampling', type=int, default=1)
#     parser.add_argument('--parallel_sampling', type=int, default=1)

#     parser.add_argument('--n_layers', type=int, default=12)
#     parser.add_argument('--hidden_dim', type=int, default=256)
#     parser.add_argument('--sparse_factor', type=int, default=-1)
#     parser.add_argument('--aggregation', type=str, default='sum')
#     parser.add_argument('--two_opt_iterations', type=int, default=1000)
#     parser.add_argument('--save_numpy_heatmap', action='store_true')

#     parser.add_argument('--project_name', type=str, default='')
#     parser.add_argument('--wandb_entity', type=str, default='')
#     parser.add_argument('--wandb_project_name', type=str, default='diffu')
#     parser.add_argument('--use_wandb', action='store_true')
#     parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
#     parser.add_argument("--use_env", default=True)

    
#     if cado_developer1:
#         parser.add_argument('--ckpt_path', type=str, default='difusco/difusco_ckpts/tsp1000_categorical.ckpt') # cado_developer1    
#     else:
#         parser.add_argument('--ckpt_path', type=str, default='/workspace/cado_developer2/project/pcb/diffusion/reward-guided-difusco/tsp50_categorical.ckpt')
    
    
#     parser.add_argument('--resume_weight_only', action='store_true')
#     parser.add_argument('--do_train', action='store_true')
#     parser.add_argument('--do_test', action='store_true')
#     parser.add_argument('--do_valid_only', action='store_true')
#     parser.add_argument('--use_classifier', action='store_true')
#     parser.add_argument('--condition', action='store_true')
#     parser.add_argument('--return_condition', action='store_true')
#     parser.add_argument('--condition_guidance_w', type=float, default=1.2)
#     # parser.add_argument('--seed', type=int, default=15)
#     parser.add_argument('--train_unconditional', type=int, default=0, help="train the unconditional prediction either")
#     parser.add_argument('--condition_dropout', type=str, default='0.1', help="train the unconditional prediction either")
#     parser.add_argument('--cost_category', type=int, default=0, help="1: cost categoriy, 0: just normal normlize")
#     parser.add_argument('--target_opt_value', type=int, default=0, help="1: return the exact optimal value, 0: just supposed value")
#     parser.add_argument('--two_opt_target', type=int, default=0, help="0: no use 2-opt target, 1: use 2-opt target") # not used 1 yet
#     parser.add_argument('--relabel_epoch', type=int, default=0,help="when >0 change cost in dataset during training")
#     parser.add_argument('--optimality_dropout', action='store_true')
#     parser.add_argument('--optimality_dropout_rate', type=float, default=0.9)

#     args = parser.parse_args()
#     return args
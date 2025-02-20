import time
import torch
import numpy as np
from ddpo_pytorch.diffusers_patch.difusco_logprob import test_eval_co, test_eval_mis



import logging

def setup_logging(log_file='evaluation.log'):
    # 로거 생성
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 파일 핸들러 생성
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 포맷터 생성
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러를 로거에 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



def format_log_message(*args, **kwargs):
    # Convert all positional arguments to strings and join them
    message = ' '.join(str(arg) for arg in args)
    
    # If there are keyword arguments, add them to the message
    if kwargs:
        message += ' ' + ' '.join(f'{k}={v}' for k, v in kwargs.items())
    
    return message

def evaluate_test(model,model_diffusion,model_args,test_dataloader, _, accelerator, global_step, sparse, use_env, inference, reward_2opt):
    """
    """
    solution_costs = []
    old_costs = []
    all_sl_loss = []
    model.eval()
    count = 0
    with torch.no_grad():
        time1 = time.perf_counter()
        for batch in test_dataloader :
            if model_args.task=='tsp':
                solved_cost, old_cost= test_eval_co(model,model_diffusion,model_args,batch,inference,use_env,sparse)
            elif model_args.task=='mis_sat' or model_args.task=='mis_er':
                solved_cost, old_cost = test_eval_mis(model, model_diffusion, model_args, batch, inference, sparse, decode_heatmap=model_args.reward_2opt)

            if model_args.print_sl_loss:
                sl_loss = model.categorical_training_step(batch, 0).detach()
            solution_costs += solved_cost
            # print('solved_cost',solved_cost)
            # print(old_cost)
            old_costs += old_cost
            # all_wo_2opt_costs += wo_2opt_costs
            # all_solution_costs += solution_costs
            # print('sl_loss',sl_loss.to('cpu').numpy())
            if model_args.print_sl_loss:
                all_sl_loss += [sl_loss]
            count += 1
            # print('gt_costs,wo_2opt_costs,solution_costs',gt_costs,wo_2opt_costs,solution_costs)
        
    log_cost = {
                "solved_cost": np.mean(solution_costs),
                "old_cost": np.mean(old_costs)
            }
    if model_args.print_sl_loss:
        log_cost['test_supervised_loss'] = torch.tensor(all_sl_loss).mean()

    if accelerator.is_main_process:
        print('time', time.perf_counter()-time1, log_cost, count)
    # if not print_log:
    # accelerator.log(
    #         log_cost,
    #         step=global_step,
    #     )
    # if reward_2opt:
    # print('solution_costs',solution_costs)
    return np.mean(solution_costs)
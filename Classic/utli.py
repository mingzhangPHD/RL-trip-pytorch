# from tensorboardX import SummaryWriter
import time
import config
import pandas as pd
import os
import numpy as np
import torch

# writer = SummaryWriter()
writer = []

results4loss = {
    'value_loss': 0.0,
    'ppo_loss': 0.0,
    'entropy': 0.0,
    'loss': 0.0
}

results4train = {
    'Train reward': 0.0,
    'Train steps':0,
    'Window Train reward':0.0
}


results4evaluate = {
    'Evaluate mean reward': 0.0,
    'Evaluate mean steps': 0,
    'Window Evaluate reward': 0.0
}

count_model = 0

def recordLossResults(results, time_step):
    results4loss['value_loss'] = results[0]
    results4loss['ppo_loss'] = results[1]
    results4loss['entropy'] = results[2]
    results4loss['loss'] = results[3]

    write2tensorboard(results=results4loss, time_step=time_step)

def recordTrainResults(results, time_step):
    results4train['Train reward'] = results[0]
    results4train['Train steps'] = results[1]
    results4train['Window Train reward'] = results[2]

    write2tensorboard(results=results4train, time_step=time_step)

def recordEvaluateResults(results, time_step):
    results4evaluate['Evaluate mean reward'] = results[0]
    results4evaluate['Evaluate mean steps'] = results[1]
    results4evaluate['Window Evaluate reward'] = results[2]

    write2tensorboard(results=results4evaluate, time_step=time_step)

def write2tensorboard(results, time_step):
    titles = results.keys()
    for title in titles:
        writer.add_scalar(title, results[title], time_step)

def store_results(evaluations, number_of_timesteps, cl_args, S_time=0, E_time=3600):
    """Store the results of a run.

    Args:
        evaluations:
        number_of_timesteps (int):
        loss_aggregate (str): The name of the loss aggregation used. (sum or mean)
        loss_function (str): The name of the loss function used.

    Returns:
        None
    """

    df = pd.DataFrame.from_records(evaluations)
    number_of_trajectories = len(evaluations[0]) - 1
    columns = ["reward_{}".format(i) for i in range(number_of_trajectories)]
    columns.append("timestep")
    df.columns = columns

    # timestamp = time.time()
    timestamp = np.around((E_time-S_time)/3600,2)
    results_fname = '{}_{}_{}_tsteps_{}_results.csv'.format(cl_args.algo_id,cl_args.env_id,
                                                                           number_of_timesteps, timestamp)
    df.to_csv(str(config.results_dir / results_fname))


def Save_model_dir(algo_id, env_id):
    fname = '{}_{}'.format(algo_id, env_id)
    # dir_rela = os.path.join(config.trained_model_dir_rela,fname)
    dir_abs = config.trained_model_dir/fname

    if not dir_abs.is_dir():
        dir_abs.mkdir()

    return dir_abs

def Save_trained_model(count, num, model, model_dir, stop_condition, reward_window4Train, reward_window4Evaluate):

    if np.mean(reward_window4Train) >= stop_condition:
        count += 1
        model_fname = 'Trained_model_{}.pt'.format(count)
        path = os.path.join(model_dir, model_fname)
        torch.save(model.state_dict(), path)
        print('********************************************************')
        print('Save model Train mean reward {:.2f}'.format(np.mean(reward_window4Train)))
        print('Save model Evaluate mean reward {:.2f}'.format(np.mean(reward_window4Evaluate)))
        print('********************************************************')
        if count >= num:
            count=0

    return count

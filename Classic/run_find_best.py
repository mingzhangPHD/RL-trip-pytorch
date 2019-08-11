'''
Find best hyper-parameters
'''

import argparse, itertools, math
import numpy as np
import torch
import importlib

np.random.seed(0)  # set random seed for keras neural networks

# define globals
algorithms = ['ppo', 'td3']
# environments = ['Acrobot', 'CartPole', 'MountainCar', 'MountainCarContinuous', 'Pendulum']
environments = ['MountainCar', 'MountainCarContinuous', 'Pendulum']
params = {
    "buffer_size": [128, 256, 512],
    "batch_size": [32, 64, 128],
    "hidden_size":[128, 128, 128],
    "learning_rate":[3e-4, 2e-4, 1e-4]
}



def parseargs():  # handle user arguments
    parser = argparse.ArgumentParser(
        description='Run NN or xgboost on Metaphlan or kmer features.')
    parser.add_argument('--disease', default='t2d', choices=['obesity', 't2d'],  # , 'wt2d', 'wt2d_10folds'],
                        help='Which disease to analyze')
    parser.add_argument('--grid_search', default='comprehensive',
                        choices=['none', 'small', 'comprehensive'],
                        help='Amount of grid search. Choices: none, small, comprehensive.')
    parser.add_argument('--seed_search', action='store_true',
                        help='Whether to search across random seeds.')
    args = parser.parse_args()
    return args




# find best hyperparameters per feature type for each classifier via grid search
def find_best_params(args, algorithms, environments, params):

    for algo in algorithms:
        for env in environments:
            mission_name = "run_"+algo+"_"+env
            mission = importlib.import_module(mission_name)
            cl_args = mission.argsparser()
            run = mission.RUN(cl_args=cl_args)
            run.train()


def run_all_default_params(algorithms, environments):

    for algo in algorithms:
        for env in environments:
            mission_name = "run_"+algo+"_"+env
            mission = importlib.import_module(mission_name)
            mission.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
            cl_args = mission.argsparser()
            run = mission.RUN(cl_args=cl_args)
            run.train()







# def print_best(best, feature_type, param_names):
#     print('\n\n\nBest parameters and results for each classifier for ' +
#           str(feature_type) + ' features...')
#     for clf in classifiers:
#         clf_param_names = param_names[clf]
#         best_params = list(zip(clf_param_names, best[clf][1]))
#         print(clf, 'best parameters:', str(best_params))
#         for m in range(len(metric_names)):
#             print(metric_names[m] + ' [mean, SD, variance] : ' + str(best[clf][0][m]))
#         print('\n')  # spacing between runs


def main():
    args = parseargs()  # get user arguments, set up experiment parameters

    run_all_default_params(algorithms = algorithms, environments = environments)
    # find_best_params(args = args, algorithms = algorithms, environments = environments, params = params)



if __name__ == '__main__':
    main()
#

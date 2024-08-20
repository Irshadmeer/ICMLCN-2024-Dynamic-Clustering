import os.path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, SAC, PPO, DDPG
import pickle
from environment import BasicScenario
from baseline import power_alo_for_closest_bs,power_alo_from_all_bs, fixed_clustering
from util import get_algorithm_function


#Random Location of the BSs
bs_locations = np.array([[0.4, 0.6, 25],
    [1.2, 0.8,25],
    [2.1 , 1.3,25],
    [0.9, 1.7,25],
    [2.3, 2.0,25],
    [1.1, 2.4,25],
    [2.0, 0.5,25],
    [0.8, 2.9,25],
    [0.6, 0.4,25],
    [0.8, 1.2,25],
    [1.3, 2.1,25], 
    [1.7, 0.9,25],
    [2.0, 2.3,25],
    [2.4, 1.1,25],
    [0.5, 2.0,25], 
    [2.9, 0.8,25]])



ALGORITHMS = {"Closest": power_alo_for_closest_bs,
              "Fixed Cluster": fixed_clustering,
              "COMP": power_alo_from_all_bs,
             }

def main(model_path, test_version, total_episodes=1000, plot: bool = True, **kwargs):

   
    scaled_bs_locations = np.copy(bs_locations)
    scaled_bs_locations[:, :2] *= 1000
    env = BasicScenario(**kwargs, bs_locations=scaled_bs_locations)

    for model_file in model_path:
        _modelname = os.path.basename(model_file)
        _algorithm = os.path.basename(model_file).split("_", 1)[0]
        func_algorithm = get_algorithm_function(_algorithm)
        model = func_algorithm.load(model_file)
        ALGORITHMS[_modelname] = (lambda x, y, s, model=model: model.predict(x, state=s, deterministic=True))


    test_data = {k: [[] for _ in range(total_episodes)] for k in ALGORITHMS}

    for _name, algorithm in ALGORITHMS.items():
        print(f"Working on algorithm: {_name}")
        for episode in range(total_episodes):
            print(f"Episode {episode+1:d}/{total_episodes:d}")
            np.random.seed(episode)
            env.seed(episode)
            obs = env.reset()
            info = env.info
            _states = None
            done = False
            while not done:
                action, _states = algorithm(obs, info, _states)
                obs, reward, done, info = env.step(action)
                test_data[_name][episode].append(info)
        file_name = f"test_data_{_name}.pickle"
        
        pickle_directory = os.path.join("pickle_files", f"pickle_files_v{test_version}")
        if not os.path.exists(pickle_directory):
            os.makedirs(pickle_directory)
        file_path = os.path.join(pickle_directory, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(test_data, f)        

    return



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", nargs="+", default=["./models/models_v6/DDPG_model.zip"]) #trained model location
    parser.add_argument("-n", "--num_users", type=int, default=6)
    parser.add_argument("-au", "--active_num_users", type=int, default=3)
    parser.add_argument("-b", "--num_base_stations", type=int, default=16)
    parser.add_argument("-x", "--max_coordinates", nargs=3, type=float, default=[3000, 3000, 120])
    parser.add_argument("-p", "--max_power_db", type=float, default=23)
    parser.add_argument("-e", "--reliability_constraint", type=float, default=.001)
    parser.add_argument("-f", "--freq", type=float, default=2.4e9)
    parser.add_argument("-s", "--rec_sensitivity", type=float, default=-5)
    parser.add_argument("-t", "--total_episodes", type=int, default=100)
    parser.add_argument("-an", "--num_antennas", type=int, default=16)
    parser.add_argument("-v", "--test_version", type=int, default=100)
    args = vars(parser.parse_args())
    main(**args)
   

#Running this test file 
#python3 test.py -m ./models/models_v2/SAC_model.zip  ./models/models_v2/DDPG_model.zip -v 3

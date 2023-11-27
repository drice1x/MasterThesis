import ray
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from gymnasium.wrappers import StepAPICompatibility
#import custom_env1
#import custom_env2
#import custom_env3
import custom_envPPO3
import custom_envPPO2
import custom_envPPO1

import torch

#for i in range(1,4):
choose_env = 3
ray.init()
if choose_env == 1:
    register_env("env1", lambda config: StepAPICompatibility(custom_envPPO1.CustomEnv1()))
    config = {"env": "env1"}
elif choose_env == 2:
    register_env("env2", lambda config: StepAPICompatibility(custom_envPPO2.CustomEnv2()))
    config = {"env": "env2"}
elif choose_env == 3:
    register_env("env3", lambda config: StepAPICompatibility(custom_envPPO3.CustomEnv3()))
    config = {"env": "env3"}
stop={"training_iteration": 10}

# train the agent
results = tune.run("PPO", config=config, stop=stop, checkpoint_freq=1, checkpoint_at_end = True,
                    local_dir='/home/paperspace/decision-diffuser/code/analysis/custom_dataset/newReward/ppo_envrechtsRUNTER'+str(choose_env))


last_checkpoint = results.get_last_checkpoint(trial=results.get_best_trial(metric="episode_reward_mean", mode="max"))


print("Hallo Hier ist der Pfad zum Checkpoint:" + str(last_checkpoint.path))
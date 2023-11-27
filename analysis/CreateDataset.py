import h5py
import gym
import numpy as np
from tqdm import tqdm
import custom_envPPO1
import custom_envPPO2
import autoencoder2
import torch
import VAE
import gymnasium as gymm
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from gymnasium.wrappers import StepAPICompatibility
#import code.analysis.custom_envPPOEx1 as custom_envPPOEx1
#import code.analysis.custom_envPPO as custom_envPPO
import custom_envPPO2
import custom_envPPO3
import custom_envPPO2rob
import custom_envPPO1rob
import custom_envPPO3rob
import torch


def create_d4rl_dataset(env_name, num_trajectories, save_path, trained_model, include_goal=True):
    
    # Create the Gym environment
    env = gymm.make(env_name)
     
    # Get the shape of the observation space
    observation_shape = env.observation_space.shape

    # Initialize the HDF5 file for saving the dataset
    with h5py.File(save_path, 'w') as hf:
        
        observation = env.reset()

        # Create datasets for observations, actions, rewards, and terminals
        obs_dataset = hf.create_dataset('observations',
                                            shape=(0,) + observation_shape,
                                            maxshape=(None,) + observation_shape,
                                            #dtype=observation.dtype) # achtung vorher: observation[0].dtype
                                            dtype = observation[0].dtype)
        #if not include_goal:
        #    real_obs_dataset =[]
        action_dataset = hf.create_dataset('actions',
                                            shape=(0,) + env.action_space.shape,
                                            maxshape=(None,) + env.action_space.shape)
        reward_dataset = hf.create_dataset('rewards',
                                            shape=(0,),
                                            maxshape=(None,),
                                            dtype=float)
        terminal_dataset = hf.create_dataset('terminals',
                                                shape=(0,),
                                                maxshape=(None,),
                                                dtype=bool)
        next_observations_dataset = hf.create_dataset('next_observations',
                                            shape=(0,) + observation_shape,
                                            maxshape=(None,) + observation_shape,
                                            #dtype=observation.dtype) # achtung vorher: observation[0].dtype
                                            dtype = observation[0].dtype)
        timeouts_dataset = hf.create_dataset('timeouts',
                                                shape=(0,),
                                                maxshape=(None,),
                                                dtype=bool)
        episodeReward = hf.create_dataset('epiReward',
                                                shape=(0,),
                                                maxshape=(None,),
                                                dtype=float)
        skillID = hf.create_dataset('skill',
                                                shape=(0,),
                                                maxshape=(None,),
                                                dtype=bool)
        
        
        for traj_idx in tqdm(range(num_trajectories)):
            # Reset the environment at the beginning of each trajectory
            observation  = env.reset()[0]
            
            
            episode_reward = 0.0
            epirewards = []
            
            
            done = False
            t_a = 0
            while not done:
                
                #obsclean= observation
                obsclean = np.where(observation == 0.5, 0, observation)
                #obsclean[obsclean == -0.33333333] = -1
                # Record the current observation
                obs_dataset.resize(obs_dataset.shape[0] + 1, axis=0)
                #obs_dataset[-1] = np.array(observation)
                obs_dataset[-1] = np.array(obsclean)
                # Choose a action suggested by the model
                #action = trained_model.compute_single_action(obs_dataset[-1])
                action = trained_model.compute_single_action(np.array(observation))
                #if traj_idx >= 495:
                #    action = env.action_space.sample()                
                # Record the chosen action
                action_dataset.resize(action_dataset.shape[0] + 1, axis=0)
                action_dataset[-1] = np.array(action)

                # Execute the action and observe the next state, reward, and termination
                next_observation, reward, done, truncated, _ = env.step(action)
                
                
                #Calculating Episode Reward as a cumulative Sum and adding to dataset
                next_observationclean = next_observation
                next_observationclean[next_observationclean == 0.5] = 0
                next_observationclean = np.where(next_observation == 0.5, 0, next_observation)
                epirewards.append(reward)
                episode_reward = sum (epirewards)
                episodeReward.resize(episodeReward.shape[0] + 1, axis=0)
                episodeReward[-1] = episode_reward                
                
        
                #next_observations_dataset.resize(next_observations_dataset.shape[0] + 1, axis=0)
                #next_observations_dataset[-1] = np.array(next_observation[0])

                next_observations_dataset.resize((obs_dataset.shape[0] + 1,) + obs_dataset.shape[1:])
                #next_observations_dataset[-1] = np.asarray(next_observation)
                next_observations_dataset[-1] = np.asarray(next_observationclean)

                # Record the reward and terminal information
                # Terminal == Truncated trajectory
                reward_dataset.resize(reward_dataset.shape[0] + 1, axis=0)
                reward_dataset[-1] = reward
                terminal_dataset.resize(terminal_dataset.shape[0] + 1, axis=0)
                terminal_dataset[-1] = done
                timeouts_dataset.resize(timeouts_dataset.shape[0] + 1, axis=0)
                timeouts_dataset[-1] = t_a > 200 #aaron
                
                
                
                #Adding Skill to Dataset
                '''
                if Done because of "reached Goal" then mark the trajectory as successfull Skill demonstration
                '''
                skillID.resize(skillID.shape[0] + 1, axis=0)
                skillID[-1] = done
                

                # Update the current observation
                observation = next_observation
                t_a +=1

    print(f"Dataset created and saved to: {save_path}")

########################################################


environmentSkill = "Env3_rechtsrunter121123"
env = 3
if env == 1:
    gymm.envs.register(id='env1', entry_point='custom_envPPO1:CustomEnv1')
    env_name_n = 'env1'
    register_env("env1", lambda config: StepAPICompatibility(custom_envPPO1.CustomEnv1()))
if env == 11:
    gym.envs.register(id='env1rob', entry_point='my_custom_env.custom_env1robust:CustomEnv1')
    env_name_n = 'env1rob'
    register_env("env1rob", lambda config: StepAPICompatibility(custom_envPPO1rob.CustomEnv1()))
if env == 2:
    gymm.envs.register(id='env2', entry_point='custom_envPPO2:CustomEnv2')
    env_name_n = 'env2'
    register_env("env2", lambda config: StepAPICompatibility(custom_envPPO2.CustomEnv2()))
if env == 21:
    gym.envs.register(id='env2rob', entry_point='my_custom_env.custom_env2robust:CustomEnv2')
    env_name_n = 'env2rob'
    register_env("env2rob", lambda config: StepAPICompatibility(custom_envPPO2rob.CustomEnv2()))
if env == 3:
    gymm.envs.register(id='env3', entry_point='custom_envPPO3:CustomEnv3')
    env_name_n = 'env3'
    register_env("env3", lambda config: StepAPICompatibility(custom_envPPO3.CustomEnv3()))
if env == 31:
    gym.envs.register(id='env3rob', entry_point='my_custom_env.custom_env3robust:CustomEnv3')
    env_name_n = 'env3rob'
    register_env("env3rob", lambda config: StepAPICompatibility(custom_envPPO3rob.CustomEnv3()))


env_names = [env_name_n]
#gym.envs.register(id='your_custom_env-v0', entry_point='my_custom_env.custom_env:CustomEnv') #aaron
#env_name = 'your_custom_env-v1'  # Replace with your desired Gym environment
save_path = '/home/paperspace/decision-diffuser/code/analysis/newdataset/withoutGoal/my_dataset'+environmentSkill+'.hdf5'  # Path to save the dataset
num_trajectories = 500

# Load the Trained Offline Policy ----------------------------------------------------- l
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from gymnasium.wrappers import StepAPICompatibility
#import code.analysis.custom_envPPOEx1 as custom_envPPOEx1
#import code.analysis.custom_envPPO as custom_envPPO
import custom_envPPO2

checkpoint_path = "/home/paperspace/decision-diffuser/code/analysis/custom_dataset/newReward/ppo_envrechtsRUNTER3/PPO/PPO_env3_26347_00000_0_2023-11-12_09-15-59/checkpoint_000010"


#Checkpoint mit -1,1 observation
#04.10.23 env2
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env2/PPO/PPO_env2_61cf0_00000_0_2023-10-04_08-55-04/checkpoint_000075


#Checkpoint mit -1,1 obs
#env1
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env1/PPO/PPO_env1_75f12_00000_0_2023-09-30_03-31-05/checkpoint_000050

#Checkpoint mit -1,1 obs
# env3
# 05.10.23
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env3/PPO/PPO_env3_94886_00000_0_2023-10-05_07-58-01/checkpoint_000075

#########
#checkpoint mit -1,0,1 observation 
#Experiment 1
##
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/withgoal/ppo_env1/PPO/PPO_env1_1050c_00000_0_2023-10-03_01-58-47/checkpoint_000075
##### Experiment2
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/withgoal/ppo_env2/PPO/PPO_env2_a55d4_00000_0_2023-10-07_02-55-29/checkpoint_000100
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/withgoal/ppo_env1/PPO/PPO_env2_527aa_00000_0_2023-10-03_02-29-16/checkpoint_000075


#04.10.23
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env2/PPO/PPO_env2_e7a58_00000_0_2023-10-04_08-01-32/checkpoint_000075

###
#Experiment3
#
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/withgoal/ppo_env3/PPO/PPO_env3_7f9cc_00000_0_2023-10-03_02-59-10/checkpoint_000075


########
#Checkpoints for
#ENVIRONMENT1 - EXPERIMENT1
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env1/PPO/PPO_env1_75f12_00000_0_2023-09-30_03-31-05/checkpoint_000100
#######
'''
###############################################################################################
Dokumentation von PPO Checkpoints mit jeweiligen Eigentschaften / Environment Configurationen
###############################################################################################

### GOOD#####
#"/home/paperspace/ray_results/PPO/PPO_custom_env_7f14b_00000_0_2023-08-23_01-10-18/checkpoint_000075"#continous observationsspace

#/home/paperspace/ray_results/PPO/PPO_custom_env_316ca_00000_0_2023-08-29_00-32-22/checkpoint_000040 #discrete actionspace, fullgrid with values 10,20,30 - unnormalized

#"/home/paperspace/ray_results/PPO/PPO_custom_env_5aa36_00000_0_2023-08-24_04-42-50/checkpoint_000075" # full grid again with values 10,20,30 - unnormalized

#"/home/paperspace/ray_results/PPO/PPO_custom_env_a109a_00000_0_2023-08-29_01-32-46/checkpoint_000050" discreteactions,normalisierteObs [0,1]

# "/home/paperspace/ray_results/PPO/PPO_custom_env_58fb2_00000_0_2023-08-31_06-14-42/checkpoint_000050"  #discreteactions,unnormalisierteObs [0,1]

#"/home/paperspace/ray_results/PPO/PPO_custom_env_58fb2_00000_0_2023-08-31_06-14-42/checkpoint_000050" #discreteactions,unnormalisierteObs [0,1]
#"/home/paperspace/ray_results/PPO/PPO_custom_env_72a5f_00000_0_2023-09-05_05-33-42/checkpoint_000050" #5x5 discrete, 4 actions,normalisiert [-1,1]
##############
'''

#########################################
# Experiment1
###############################
#"/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env1/PPO/PPO_env1_75f12_00000_0_2023-09-30_03-31-05/checkpoint_000100"
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env2/PPO/PPO_env1_d6305_00000_0_2023-09-19_05-19-53/checkpoint_000120   ( mit Goal in obs und actino Discrte4)

# action 8, obs -1,1
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env2/PPO/PPO_env1_0513b_00000_0_2023-09-25_02-58-04/checkpoint_000100
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env1/PPO/PPO_env1_de25e_00000_0_2023-09-25_05-34-28/checkpoint_000075

# obs mit (-1,1)
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env2/PPO/PPO_env1_60f49_00000_0_2023-09-23_02-20-04/checkpoint_000110
#/home/paperspace/decision-diffuser/code/analysis/custom_dataset/ppo_env2/PPO/PPO_env1_4b801_00000_0_2023-09-24_03-29-51/checkpoint_000200

##############
ray.init()
 # ändern: kein Gymnasium! gym vernwenden!!!
trained_model = Algorithm.from_checkpoint(checkpoint_path)
# ---------------------------------------------------------------------------- 



create_d4rl_dataset(env_name_n, num_trajectories, save_path, trained_model, include_goal=True)

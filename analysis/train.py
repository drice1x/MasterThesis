import sys
sys.path.append(':/home/paperspace/decision-diffuser/code:/home/paperspace/decision-diffuser/code/analysis:/home/paperspace/decision-diffuser/code/diffuser:/home/paperspace/decision-diffuser/code/scripts')

sys.path.append('$PYTHONPATH:/home/paperspace/decision-diffuser/code/analysis/')

import os

os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/home/paperspace/.mujoco/mujoco200/bin"

import os

os.chdir('/home/paperspace/decision-diffuser/code/analysis/')

print("pfad:")

print( os.getcwd())  # Prints the current working directory

print("-----")

import sys

print(sys.path)
import torch

if __name__ == '__main__':
    import os
    import sys
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.train import main
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep
    
    # das muss einfach am anfang stehen
    # Register your custom environment
    import gym
    #gym.envs.register(id='your_custom_env-v0', entry_point='my_custom_env.custom_env:CustomEnv') #aaron
    env = 2
    if env == 1:
        gym.envs.register(id='env1', entry_point='my_custom_env.custom_env1:CustomEnv1')
        env_name_n = 'env1'
        gym.envs.register(id='env1mix', entry_point='my_custom_env.custom_env1mix:CustomEnv1')
        env_name_n = 'env1mix'
    if env == 11:
        gym.envs.register(id='env1mix', entry_point='my_custom_env.custom_env1mix:CustomEnv1')
        env_name_n = 'env1mix'
        #register_env("env1", lambda config: StepAPICompatibility(custom_env1.CustomEnv1()))
    if env == 2:
        gym.envs.register(id='env1', entry_point='my_custom_env.custom_env1:CustomEnv1')
        env_name_n = 'env1'
        gym.envs.register(id='env1mix', entry_point='my_custom_env.custom_env1mix:CustomEnv1')
        env_name_n = 'env1mix'
        gym.envs.register(id='env2', entry_point='my_custom_env.custom_env2:CustomEnv2')
        env_name_n = 'env2'
        gym.envs.register(id='env2mix', entry_point='my_custom_env.custom_env2mix:CustomEnv2')
        env_name_n = 'env2mix'
        gym.envs.register(id='env3', entry_point='my_custom_env.custom_env3:CustomEnv3')
        env_name_n = 'env3'
        gym.envs.register(id='env3mix', entry_point='my_custom_env.custom_env3mix:CustomEnv3')
        env_name_n = 'env3mix'
    if env == 21:
        gym.envs.register(id='env2mix', entry_point='my_custom_env.custom_env2mix:CustomEnv2')
        env_name_n = 'env2mix'
        #register_env("env2", lambda config: StepAPICompatibility(custom_env2.CustomEnv2()))
    if env == 3:
        gym.envs.register(id='env2', entry_point='my_custom_env.custom_env2:CustomEnv2')
        env_name_n = 'env2'
        gym.envs.register(id='env2mix', entry_point='my_custom_env.custom_env2mix:CustomEnv2')
        env_name_n = 'env2mix'
        gym.envs.register(id='env3', entry_point='my_custom_env.custom_env3:CustomEnv3')
        env_name_n = 'env3'
        gym.envs.register(id='env3mix', entry_point='my_custom_env.custom_env3mix:CustomEnv3')
        env_name_n = 'env3mix'
    if env == 31:
        gym.envs.register(id='env3mix', entry_point='my_custom_env.custom_env3mix:CustomEnv3')
        env_name_n = 'env3mix'
        #register_env("env3", lambda config: custom_env3.CustomEnv3())

    sweep = Sweep(RUN, Config).load("default_inv.jsonl") # bullet-hopper-medium-replay-v0


    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()

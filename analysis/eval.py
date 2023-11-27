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
from tqdm import tqdm

print(sys.path)
if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.evaluate_inv_parallel import evaluate
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep

    # das muss einfach am anfang stehen
    # Register your custom environment
    import gym
    env = 2
    if env == 1:
        gym.envs.register(id='env1', entry_point='my_custom_env.custom_env1:CustomEnv1')
        env_name_n = 'env1'
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
        gym.envs.register(id='env1rob', entry_point='my_custom_env.custom_env1robust:CustomEnv1')
        env_name_n = 'env1rob'
        gym.envs.register(id='env1robmix', entry_point='my_custom_env.custom_env1mixrobust:CustomEnv1')
        env_name_n = 'env1robmix'
        gym.envs.register(id='env2rob', entry_point='my_custom_env.custom_env2robust:CustomEnv2')
        env_name_n = 'env2rob'
        gym.envs.register(id='env2robmix', entry_point='my_custom_env.custom_env2mixrobust:CustomEnv2')
        env_name_n = 'env2robmix'
        gym.envs.register(id='env3rob', entry_point='my_custom_env.custom_env3robust:CustomEnv3')
        env_name_n = 'env3rob'
        gym.envs.register(id='env3robmix', entry_point='my_custom_env.custom_env3mixrobust:CustomEnv3')
        env_name_n = 'env3robmix'
        #register_env("env2", lambda config: StepAPICompatibility(custom_env2.CustomEnv2()))
    if env == 3:
        gym.envs.register(id='env3', entry_point='my_custom_env.custom_env3:CustomEnv3')
        env_name_n = 'env3'
        gym.envs.register(id='env3mix', entry_point='my_custom_env.custom_env3mix:CustomEnv3')
        env_name_n = 'env3mix'
        #register_env("env3", lambda config: custom_env3.CustomEnv3())

    #gym.envs.register(id='your_custom_env-v0', entry_point='my_custom_env.custom_env:CustomEnv') #aaron
    sweep = Sweep(RUN, Config).load("default_inv.jsonl") # bullet-hopper-medium-replay-v0

    for kwargs in tqdm(sweep): # kann über versch. konfigurationen sweepen
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(evaluate, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()
import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os
import gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
from analysis.evaluate_dataset import determine_action, get_actions, estimate_actions
import autoencoder2
from analysis.evaluate_dataset import  gradient_action, round_values2,round_values1,round_values, find_position_with_value, calculate_action1, calculate_action
import matplotlib.pyplot as plt
import csv

model_path = '/home/paperspace/decision-diffuser/code/autoencoder/newae/autoencoderEnv2.pth'
device = torch.device("cpu")
#pretrained_autoencoder.to('cuda:0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_autoencoder = autoencoder2.Autoencoder().to('cuda:0')
pretrained_autoencoder.load_state_dict(torch.load(model_path, map_location= device))
#pretrained_autoencoder= 1 #BLUFF LÖSCHEN


def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{self.step}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt') #aaron
    
    state_dict = torch.load(loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )

    dataset = dataset_config()
    
    print("type dataset:", type(dataset))
    print("keys dataset: ", dataset.fields.keys) ##aaron bugfix version
    
    
    def make_latent_evaluate(key, autoencoder = pretrained_autoencoder):
        # Get the shapes of the dataset
        num_entries, num_samples, observation_dim = dataset.fields[key].shape

        # Initialize a new array to hold encoded observations
        encoded_observations = np.empty((num_entries, num_samples, 16))

        # Loop through each entry
        for entry_idx in range(num_entries):
            for sample_idx in range(num_samples):
                # Get the observations for the current entry and sample
                observations_entry_sample = dataset.fields[key][entry_idx, sample_idx]
                
                # Fit normalization
                #autoencoder.fit_normalization(observations_entry_sample)
                
                # Convert to PyTorch tensor
                #observations_tensor = torch.tensor(autoencoder.normalize_data(observations_entry_sample), dtype=torch.float64)
                observations_tensor = torch.tensor(observations_entry_sample, dtype=torch.float64)
                # Encode the observations using the autoencoder
                encoded_obs_sample = autoencoder.encoder(observations_tensor).cpu().detach().numpy()

                # Store the encoded observations for the current entry and sample
                encoded_observations[entry_idx, sample_idx] = encoded_obs_sample

        return encoded_observations
    
    
    
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
        
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        #hidden_dim=Config.hidden_dim,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)
    logger.print(utils.report_parameters(model), color='green')
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])
    horizon1 = diffusion_config.horizon -1

    num_eval = 200
    device = Config.device

    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]  
    
    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)

    t = 0
    def encode_obs_targets (obs_tensor, target_tensor):
        encoded_obs = []
        for entry in obs_tensor:
            # Ensure the entry tensor has the correct shape (64 for the input dimension)
            #entry = entry.view(1, -1)
            entry = entry.to('cuda:0')  # Move input data to the same GPU
            encoded_entry = pretrained_autoencoder.encode(entry)
            encoded_obs.append(encoded_entry.squeeze().detach().cpu().numpy())
            
            
        encoded_target = []
        for target in target_tensor:
            # Ensure the entry tensor has the correct shape (64 for the input dimension)
            #target = target.view(1, -1)
            target = target.to('cuda:0')  # Move input data to the same GPU
            encoded_goal = pretrained_autoencoder.encode(target)
            encoded_target.append(encoded_goal.squeeze().detach().cpu().numpy())
        return encoded_obs, encoded_target
    
    
    obs_list = [env.reset()[None] for env in env_list]
    #obs_list = [result[0][None] for result in results]
    
    t = 0
    obs_list = [env.reset()[None] for env in env_list]
    target_list = [env.target()[None] for env in env_list]
    obs = np.concatenate(obs_list, axis=0)
    targets = np.concatenate(target_list, axis=0)
    recorded_obs = [deepcopy(obs[:, None])]
    obs_tensor = torch.tensor(obs)
    target_tensor = torch.tensor(targets)


    obs = np.concatenate(obs_list, axis=0)
    
    encoded_obs , encoded_target = encode_obs_targets (obs_tensor, target_tensor)


    history = {j: {'obs': [], 'action': [], 'sample': [] ,'reward': [], 'done': []} for j in range(num_eval)}
    latentspecs = {k: {'latent': [], 'decoded': [], 'action': []} for k in range(num_eval)}
    import pandas as pd
    while sum(dones) <  num_eval:
    #for k, p_steps in enumerate()
        #obs = np.array([np.array(normalize1(o)) for o in obs])
        #targets = np.array([np.array(normalize1(o)) for o in targets])
        #obs = dataset.normalizer.normalize(obs, 'observations')
        #targets = dataset.normalizer.normalize(targets, 'observations')
        #conditions = {0: to_torch(obs, device=device),
        #              horizon1: to_torch(targets, device=device)}
        conditions = {0: to_torch(np.array(encoded_obs), device=device),
                      horizon1: to_torch(np.array(encoded_target), device=device)}
        
        samples = trainer.ema_model.conditional_sample(conditions, returns=returns)
        
        
        #obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        #obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        #action = trainer.ema_model.inv_model(obs_comb)
        
        

        my_actions = []
        for idx, trajectory in enumerate(samples):
            actions_for_onetrajectory = []
            #for index, row in reversed(list(enumerate(trajectory))):
            #    if not (np.abs(row) <= 0.02).all():
            #        result = row
            #        result_index = index
            #        break
            agentpos = []
            for i in range( len(trajectory)-1):
                latentspecs[idx]['latent'].append(trajectory[i])
                #if i == 0:
                #    estimatedAPos = round_values2(pretrained_autoencoder.decode(trajectory[i]))
                     #Function to find the position of "1" in the array

                #    agentpos.append(find_position_with_value(estimatedAPos.reshape((6,6)), 1))                    
                #latentspecs[idx]['latent'].append(trajectory[i])
                #latentspecs[idx]['decoded'].append(pretrained_autoencoder.decode(trajectory[i]))
                #history[j]['sample'].append(trajectory[i])
                #dec1 = pretrained_autoencoder.decode(trajectory[i])
                #decplus = pretrained_autoencoder.decode(trajectory[i+1])
                actions_for_onetrajectory.append(calculate_action1(round_values2(pretrained_autoencoder.decode(trajectory[i])),round_values2(pretrained_autoencoder.decode(trajectory[i+1]))))
                #actions_for_onetrajectory.append(calculate_action1(round_values2(trajectory[i]),round_values2(trajectory[i+1])))
                
                
                #newPos = calculate_new_position (agentpos[i], actions_for_onetrajectory[-1])
                #agentpos.append(newPos)
            my_actions.append(actions_for_onetrajectory)
            #confident_actions = estimate_actions(trajectory, proximal_steps, normalize=True)    
            #conf_actions.append(confident_actions)
        
        
        if t == 0:
            normed_observations = samples[:, :, :]
            #observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            #savepath = os.path.join('images', 'sample-planned.png')
            
        done_count = 0
        crash_count = 0
        truncated_count =0 
        obs_list = []
        episode_rewards = []
        steps_taken=[]
        collision = False
        total_reward =0
        goals_with_coll = 0
        goals_without_coll = 0
        for j in range(num_eval):
            epi_reward = 0
            collision = False
            step_count=0
            for i in range(len(my_actions[j])):
                this_obs, this_reward, this_done, _ = env_list[j].step(my_actions[j][i])
                latentspecs[j]['action'].append(my_actions[j][i])
                history[j]['obs'].append(this_obs.reshape((6,6)))
                history[j]['reward'].append(this_reward)
                history[j]['done'].append(this_done)
                history[j]['action'].append(my_actions[j][i])
                
                
                step_count+=1
                epi_reward += this_reward
                if this_reward == -10:
                    crash_count += 1
                    collision = True
                if this_reward == -30:
                    truncated_count+=1 
                    break                   
                if this_reward == 30 and not collision:
                    done_count += 1
                    goals_without_coll +=1
                    steps_taken.append(step_count)
                    break
                elif this_reward == 30 and collision:
                    done_count+=1
                    goals_with_coll+=1
                    steps_taken.append(step_count)
                    break                
                elif i==len(my_actions[j])-1:
                    steps_taken.append(step_count) 

                #if this_done:
                #    if dones[j] == 1:
                #        pass
                #    else:
                #        dones[j] = 1
                #        episode_rewards[j] = epi_reward
                #        #logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
            logger.print(f"Episode ({j}): {epi_reward:.4f}", color='green')
            episode_rewards.append(epi_reward)
        logger.print("="*60, color='blue')
        logger.print(f"Out of {num_eval} tasks {done_count} found the goal and there were {crash_count} obstacle collisions", color='blue')
        logger.print("="*60, color='blue')    
                   
        logger.print(f"Out of {num_eval} tasks there were {truncated_count} truncated trajectories", color='blue')
        logger.print(f"Mean Episode Reward of this Model: {sum(episode_rewards)/num_eval}", color='blue')
        logger.print("="*60, color='blue') 
        logger.print("="*60, color='blue')  
        logger.print(f"{goals_without_coll} goal without collision", color='blue')
        logger.print(f"{goals_with_coll} goal with collision", color='blue')
        logger.print("="*60, color='blue') 
    # Set the custom file path

         

        data = np.vstack((samples[0].cpu())).T
        
        meanvalue = np.mean(data, axis=0)
        minvalue = np.min(data, axis=0)
        maxvalue = np.max(data,axis=0)
        stddev = np.std(data, axis =0)
        
        timepoint = np.arange(len(meanvalue))
        
        plt.figure(figsize = (12,6))
        
        plt.subplot(2,2,1)
        plt.plot(timepoint, meanvalue, marker="o")
        plt.title("mean over time")
        plt.xlabel("time")
        plt.ylabel("mean")
        
        plt.subplot(2,2,2)
        plt.plot(timepoint, minvalue, marker="o")
        plt.title("min over time")
        plt.xlabel("time")
        plt.ylabel("min")
        
        plt.subplot(2,2,3)
        plt.plot(timepoint, maxvalue, marker="o")
        plt.title("max over time")
        plt.xlabel("time")
        plt.ylabel("max")   
        
        
        plt.subplot(2,2,4)
        plt.plot(timepoint, stddev, marker="o")
        plt.title("std over time")
        plt.xlabel("time")
        plt.ylabel("std")
        
        plt.tight_layout()
        plt.show()     
        
        
        break
        # Initialize empty lists to store flattened data
        '''        
        keys = []
        list_type = []
        tensor_value = []

        # Flatten the dictionary
        for key, values in latentspecs.items():
            for list_name, value_list in values.items():
                for i, value in enumerate(value_list):
                    keys.append(key)
                    list_type.append(list_name)
                    if list_name == "action":
                        tensor_value.append(value)
                    else:
                        tensor_value.append(value.cpu().detach().numpy())

        # Create a DataFrame
        df = pd.DataFrame({
            'Key': keys,
            'List_Type': list_type,
            'Tensor_Value': tensor_value
        })
        # Specify the path where you want to save the CSV file
        csv_file_path = '/home/paperspace/decision-diffuser/code/analysis/latentdata3.csv'

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)

        # Confirm that the CSV file has been created
        print(f"CSV file '{csv_file_path}' has been created.")
        # Display the DataFrame
        #print(df)
    dones1 = [0 for _ in range(num_eval)]
    episode_rewards1 = [0 for _ in range(num_eval)] 
    while sum(dones1) <  num_eval:
        conditions = {0: to_torch(np.array(encoded_obs), device=device),
                      horizon1: to_torch(np.array(encoded_target), device=device)}
        
        samples = trainer.ema_model.conditional_sample(conditions, returns=returns)
        
        
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)
        if t == 0:
            normed_observations = samples[:, :, :]
            #observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            #savepath = os.path.join('images', 'sample-planned.png')
            #renderer.composite(savepath, observations)

        obs_list = []
        for i in range(num_eval):
            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
            obs_list.append(this_obs[None])
            if this_done:
                if dones[i] == 1:
                    pass
                else:
                    dones[i] = 1
                    episode_rewards[i] += this_reward
                    #logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
            else:
                if dones[i] == 1:
                    pass
                else:
                    episode_rewards[i] += this_reward

        obs_list1=[]
        for i in range(num_eval):
            this_obs, this_reward, this_done, _ = env_list[i].step(action1[i])
            obs_list1.append(this_obs[None])
            if this_done:
                if dones1[i] == 1:
                    pass
                else:
                    dones1[i] = 1
                    episode_rewards1[i] += this_reward
                    #logger.print(f"Episode1 ({i}): {episode_rewards1[i]}", color='green')
            else:
                if dones1[i] == 1:
                    pass
                else:
                    episode_rewards1[i] += this_reward

        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1    
        
    recorded_obs = np.concatenate(recorded_obs, axis=1)
    #savepath = os.path.join('images', f'sample-executed.png')
    #renderer.composite(savepath, recorded_obs)
    episode_rewards = np.array(episode_rewards)
    episode_rewards1 = np.array(episode_rewards1)

    logger.print(f"ORIGINALaverage_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}", color='green')
    logger.log_metrics_summary({'ORIGINALaverage_ep_reward':np.mean(episode_rewards), 'std_ep_reward':np.std(episode_rewards)})
    logger.print(f"11ORIGINALaverage_ep_reward: {np.mean(episode_rewards1)}, std_ep_reward: {np.std(episode_rewards1)}", color='green')
    logger.log_metrics_summary({'11ORIGINALaverage_ep_reward':np.mean(episode_rewards1), 'std_ep_reward':np.std(episode_rewards1)})


    print("ende im gelände")
    print(latentspecs[0])
    print("schluss")
'''
from collections import namedtuple
import numpy as np
import torch
import pdb
import autoencoder2
import VAE

#

torch.set_default_dtype(torch.float64)

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


device = torch.device("cuda" )#if torch.cuda.is_available() else "cpu")

#vae = VAE.BetaVAE().to(device)
# Load the trained model state
#vae.load_state_dict(torch.load(model_path1, map_location= device))

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

model_path = '/home/paperspace/decision-diffuser/code/autoencoder/newae/autoencoderEnv2.pth'
#device = torch.device("cpu")
device = torch.device("cuda" )#if torch.cuda.is_available() else "cpu")
pretrained_autoencoder = autoencoder2.Autoencoder().to(device)
pretrained_autoencoder.load_state_dict(torch.load(model_path, map_location= device))


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=70000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=100, include_returns=True, autoencoder = pretrained_autoencoder):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.autoencoder = autoencoder
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn) # aaron: hier wird aus d4rl env ein sequenc dataset iterator gemacht.

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.fields = fields
        
        self.fields['observations'] = self.make_latent('observations')
        self.fields['next_observations'] = self.make_latent('next_observations')                                                           
        self.normalizer = DatasetNormalizer(self.fields, normalizer, path_lengths=self.fields['path_lengths'])
        self.indices = self.make_indices(self.fields.path_lengths, horizon)
        
        self.indices1 = self.make_indices1(self.fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]

        #self.indices_5 = np.where(self.fields.observations[0][0] == 5)[0]
        #self.indices_1 = np.where(self.fields.observations[0][0] == 1)[0]
        
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
                
    def make_latent(self, key, latentdim= 16):
        # Get the shapes of the dataset
        num_entries, num_samples, observation_dim = self.fields[key].shape

        # Initialize a new array to hold encoded observations
        encoded_observations = np.empty((num_entries, num_samples, latentdim))

        # Loop through each entry
        for entry_idx in range(num_entries):
            for sample_idx in range(num_samples):
                # Get the observations for the current entry and sample
                observations_entry_sample = self.fields[key][entry_idx, sample_idx]

                # Fit normalization
                #self.autoencoder.fit_normalization(observations_entry_sample)
                
                # Convert to PyTorch tensor
                #observations_tensor = torch.tensor(self.autoencoder.normalize_data(observations_entry_sample), dtype=torch.float64)
                observations_tensor = torch.tensor(observations_entry_sample, dtype=torch.float64)
                observations_tensor = observations_tensor.to('cuda')
                # Encode the observations using the autoencoder
                encoded_obs_sample = self.autoencoder.encode(observations_tensor).cpu().detach().numpy()

                # Store the encoded observations for the current entry and sample
                encoded_observations[entry_idx, sample_idx] = encoded_obs_sample

        return encoded_observations

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        # hier muss der horizon angepasst werden. es muss die indices sein bei denen eine trajektorie endet
        
        indices = []
        for i, path_length in enumerate(path_lengths): #len(path_lengths = number of trajectories)
            
            #indices_of_ones = np.where(self.fields.skill[i].flatten() == 1)
            #indices_of_ones = [np.where (self.fields.skill[0] ==1)[0]]

            max_start = min(path_length - 1, self.max_path_length - horizon) #original
            #max_start1 = min(path_length - 1, self.max_path_length - max(indices_of_ones[i]))#[i])
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon) # original
                #max_start1 = min(max_start, path_length - indices_of_ones[i])
            for k, start in enumerate(range(max_start)):
                end = start + horizon #original
                end1 = path_length #indices_of_ones[i][k]
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices
    
    def make_indices1(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        # hier muss der horizon angepasst werden. es muss die indices sein bei denen eine trajektorie endet
        
        indices = []
        for i, path_length in enumerate(path_lengths): #len(path_lengths = number of trajectories)
            
            #indices_of_ones = np.where(self.fields.skill[i].flatten() == 1)
            #indices_of_ones = [np.where (self.fields.skill[0] ==1)[0]]

            max_start = min(path_length - 1, self.max_path_length - horizon) #original
            #max_start1 = min(path_length - 1, self.max_path_length - max(indices_of_ones[i]))#[i])
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon) # original
                #max_start1 = min(max_start, path_length - indices_of_ones[i])
            for k, start in enumerate(range(max_start)):
                end = start + horizon #original
                end1 = path_length #indices_of_ones[i][k]
                indices.append((i, start, end1))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        #aaron: was wird hier an condition ausgewertet?
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        #path_ind1, start1, end1 = self.indices1[idx]
        
        
        #start = 0
        #end = self.horizon 

        #observations = self.fields.normed_observations[path_ind, start:end]
        #actions = self.fields.normed_actions[path_ind, start:end]
        observations = self.fields.observations[path_ind, start:end]
        observations1 = self.fields.observations[path_ind, start:end+1]
        actions = self.fields.actions[path_ind, start:end] 
        next_observations = self.fields.next_observations[path_ind, start:end+1]      
        
        
        # Fill NaN values with 0
        #observations = np.nan_to_num(observations, nan=0.0)
        # Fill NaN values with 0
        #actions = np.nan_to_num(actions, nan=0.0)
        
        
        #observations1 = self.fields.observations[path_ind, start1:end1]
        #actions1 = self.fields.actions[path_ind, start1:end1]      
        
        #hier np.where terminal==1 nutzen und das ende einer trajektorie markieren. dieses nutzen um "end" zu ersetzen
        

        conditions = self.get_conditions(observations, next_observations)
        #conditions1 = self.get_conditions(observations1)
        #conditions2 = self.get_conditions(observations2)
        
        trajectories = np.concatenate([actions, observations], axis=-1)
        #trajectories1 = np.concatenate([actions1, observations1], axis=-1)
        #trajectories2 = np.concatenate([actions2, observations2], axis=-1)
        

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class CondSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env) # aaron: hier wird env geladen
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        
        
        #self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def normalize_observation(self, observation):
        min_value = 0
        max_value = 30
        normalized_observation = (observation - min_value) / (max_value - min_value) 
        return normalized_observation

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        t_step = np.random.randint(0, self.horizon)
        #self.fields.observations = self.normalize_observation(self.fields.observations)

        observations = self.normalize_observation(self.fields.observations)[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]

        traj_dim = self.action_dim + self.observation_dim

        conditions = np.ones((self.horizon, 2*traj_dim)).astype(np.float32)

        # Set up conditional masking
        conditions[t_step:,:self.action_dim] = 0
        conditions[:,traj_dim:] = 0
        conditions[t_step,traj_dim:traj_dim+self.action_dim] = 1

        if t_step < self.horizon-1:
            observations[t_step+1:] = 0

        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class GoalDataset(SequenceDataset):
    
    
    def find_change_indices(array):
        changes = np.where(np.diff(array) != 0)[0] + 1
        return changes



    def get_conditions(self, observations, next_observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        '''        
        threshold = 0.01
        observations = np.array(observations)
        differences = np.abs(observations[:-1] - observations[1:])
        change_indices = np.where(np.any(differences > threshold, axis=1))[0]

        if change_indices.size > 0:
            result_index = change_indices[-1] + 1
        else:
            result_index = 0
        '''
        previous_array = None

        # Iterate through the entries
        for i in range(len(observations)):
            current_array = observations[i]
            
            if previous_array is not None:
                # Compare the current array with the previous one
                if np.array_equal(current_array, previous_array):
                    # Arrays are the same, so the change has stopped
                    index_of_last_change = i - 1
                    break
            
            # Update the previous array for the next iteration
            previous_array = current_array
        else:
            # If the loop completes without a break, all arrays are different
            index_of_last_change = len(observations) - 1
        return {
            0: observations[0],
            #index_of_last_change-1: observations[index_of_last_change-1]
            #index_of_last_change-1: observations[index_of_last_change-1],
            self.horizon - 1: observations[index_of_last_change-1]
                }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch

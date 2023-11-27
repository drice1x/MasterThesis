'''
Filtering the h5 Dataset files to get the best n trajectories

'''
import pandas as pd
import h5py
import gym
from gym import spaces
import numpy as np
import custom_envPPO1
import custom_envPPO2
import custom_envPPO3


class CustomEnv(gym.Env):
    def __init__(self, dataset_path):
        super().__init__()

        # Load the dataset
        self.dataset = h5py.File(dataset_path, 'r')

        # Extract necessary information from the dataset
        self.trajectories = list(self.dataset.keys())
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10 * 10,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)#spaces.Box(low=-1.0, high=1.0, shape=(0, 1), dtype=np.float32) #self.dataset['actions'][0].shape[0]

    def reset(self):
        # Implement your reset logic here
        pass

    def step(self, action):
        # Implement your step logic here
        pass

    def get_dataset(self):
        return self.dataset

    def close(self):
        self.dataset.close()
        
        


def AnalyseDataset (dataset_path):
    
    # Example usage
    dataset_path = '/home/paperspace/decision-diffuser/code/analysis/my_dataset22.hdf5'  # Path to your D4RL dataset

    env = CustomEnv(dataset_path)
    dataset = env.get_dataset()



    # Access dataset information
    print(dataset.keys()) 




    dfactions = pd.DataFrame(np.array(dataset['actions']),columns= ["actions"])
    dfrew = pd.DataFrame(np.array(dataset['rewards']),columns = ["rewards"])


    #dfobs = pd.DataFrame(np.array(dataset['observations']),columns = ["observations"])
    dfrew = pd.DataFrame(np.array(dataset['rewards']),columns = ["rewards"])
    dfepirew = pd.DataFrame(np.array(dataset['epiReward']),columns = ["episodeReward"])
    dftimeouts = pd.DataFrame(np.array(dataset['timeouts']),columns = ["timeouts"])
    dfdone = pd.DataFrame(np.array(dataset['skill']),columns = ["skillID"])
    dfterminals = pd.DataFrame(np.array(dataset['terminals']),columns = ["terminals"])

    dfobs = pd.DataFrame(np.array(dataset['observations']), columns=["observation_" + str(i) for i in range(25)])
    merged_df = pd.DataFrame({"Merged": dfobs.values.tolist()})


    '''
    df epirew muss hinzugefügt werden
    '''
    df = pd.concat([dfactions, dfrew, dftimeouts, dfterminals, dfdone ,dfepirew, merged_df], axis=1)
    df = df.rename(columns={"Merged": "observations"})

    episode_rewards = df.loc[df['terminals'], 'episodeReward'].tolist()
    count = df['terminals'].sum()

    meanepisodereward = sum(episode_rewards)/count

    print("Mean Episode Reward: ", meanepisodereward)
    
    return meanepisodereward, df


#code/analysis/my_datasetGOOD_500samplesEXPERIMENT1.hdf5
#code/analysis/my_datasetGOOD_500samplesEXPERIMENT1.hdf5
# Example usage
dataset_path = '/home/paperspace/decision-diffuser/code/analysis/newdataset/withoutGoal/env3-con.hdf5'  # Path to your D4RL dataset

env = CustomEnv(dataset_path)
dataset = env.get_dataset()



# Access dataset information
print(dataset.keys()) 



#dfactions = pd.DataFrame(np.array(dataset['actions']), columns=["actions" + str(i) for i in range(2)])
#mergedaction_df = pd.DataFrame({"actions": dfactions.values.tolist()})
dfactions = pd.DataFrame(np.array(dataset['actions']),columns= ["actions"])
dfrew = pd.DataFrame(np.array(dataset['rewards']),columns = ["rewards"])


#dfobs = pd.DataFrame(np.array(dataset['observations']),columns = ["observations"])
dfrew = pd.DataFrame(np.array(dataset['rewards']),columns = ["rewards"])
dfepirew = pd.DataFrame(np.array(dataset['epiReward']),columns = ["episodeReward"])
dftimeouts = pd.DataFrame(np.array(dataset['timeouts']),columns = ["timeouts"])
dfdone = pd.DataFrame(np.array(dataset['skill']),columns = ["skillID"])
dfterminals = pd.DataFrame(np.array(dataset['terminals']),columns = ["terminals"])

dfobs = pd.DataFrame(np.array(dataset['observations']), columns=["observation_" + str(i) for i in range(64)])
mergedobs_df = pd.DataFrame({"Merged": dfobs.values.tolist()})


'''
df epirew muss hinzugefügt werden
'''
df = pd.concat([dfactions, dfrew, dftimeouts, dfterminals, dfdone ,dfepirew, mergedobs_df], axis=1)
df = df.rename(columns={"Merged": "observations"})

print(df)

df = df.fillna(False)
batch_ended = df['terminals'] | df['skillID']



# Get the last episodeReward value for each batch
last_episode_rewards = df.loc[batch_ended, 'episodeReward'].tolist()


# First, create a mask to filter rows where either 'terminals' or 'skillID' is True
mask = df['terminals'] | df['skillID']

# Then, apply the mask to filter rows
filtered_d1 = df[mask]

# Now, count the occurrences where 'rewards' is equal to 100
count_occurrences = len(df[df['rewards'] == 30])

print("Number of occurrences where 'rewards' is 30 or Goals founded:", count_occurrences)

merged_column1 = df["terminals"] | df["skillID"]

# Count the number of True values in the merged column
count_true_values = merged_column1.sum()

print("Number of True values in the merged column:", count_true_values)


mean_episode_rewards = df.groupby(['terminals', 'skillID'])['episodeReward'].mean()
episode_rewards = df.loc[df['terminals'], 'episodeReward'].tolist()
#df['terminals'].sum()
#df['skillID'].sum()

count = max(df['terminals'].sum(), df['skillID'].sum())

meanepisodereward = sum(last_episode_rewards)/count

print("Mean Episode Reward: ", meanepisodereward)

df.to_json('/home/paperspace/decision-diffuser/code/analysis/data_experiment3_121123.json')
#df.to_parquet('/home/paperspace/decision-diffuser/code/analysis/data_experiment2Expert.parquet')
#df.to_json('/home/paperspace/decision-diffuser/code/analysis/data_experiment2Expert.csv')



print("--------")


# Assuming "df" is your DataFrame with the "skillID" column
last_true_index = df[df['skillID'] == True].index[-1]
columns_to_keep = df.columns[:last_true_index + 1]
print("last true index: ",last_true_index)
print("columns to keep: ",columns_to_keep)
# Extract the DataFrame with only the required columns
#result_df = df[columns_to_keep]
#print(df)

#filtered_df = df[(df['terminals'] == False) & (df['skillID'] == True)]
#print(filtered_df)

'''dfcopy = df.copy()

n_highest = 3  # Number of highest batches to retrieve


df['batch_start'] = df['terminals'].shift(1)
df['batch_end'] = df['terminals']
df['batch_reward'] = df.groupby((df['batch_start']!= df['batch_start'].shift(1)).cumsum())[
    'episodeReward'].transform('last')
df_sorted = df.sort_values(by=['batch_reward', 'batch_start'], ascending=[False, True])

batch_indices = df_sorted[df_sorted['batch_end']].index.tolist()[:n_highest]
#df[df['column_name']].index.tolist()
idx = batch_indices[-1]
df_filtered = df_sorted.loc[:idx]


print(df_filtered)
'''
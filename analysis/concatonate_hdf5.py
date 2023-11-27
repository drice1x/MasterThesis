import h5py
import numpy as np

def concatenate_hdf5(file_names, output_name):
    with h5py.File(output_name, 'w') as output_file:
        # Initialize a dictionary to hold concatenated data
        concatenated_data = {}
        # Iterate through file names
        for file_name in file_names:
            with h5py.File(file_name, 'r') as file:
                # Iterate through datasets in file
                for key in file.keys():
                    if key in concatenated_data:
                        # Concatenate the datasets
                        concatenated_data[key] = np.concatenate((concatenated_data[key], file[key][:]))
                    else:
                        concatenated_data[key] = file[key][:]
        # Write the concatenated data to the new file
        for key, data in concatenated_data.items():
            output_file.create_dataset(key, data=data)

def shuffle_trajectories(input_name, output_name):
    with h5py.File(input_name, 'r') as file:
        terminal_flags = file['terminals'][:]  # Assuming 'terminals' is the name of the dataset
        
        # Identify start and end indices of each trajectory
        trajectory_indices = np.where(terminal_flags == True)[0]
        start_indices = np.concatenate(([0], trajectory_indices[:-1] + 1))
        end_indices = trajectory_indices
        
        # Shuffle trajectories
        shuffled_indices = np.arange(len(start_indices))
        np.random.shuffle(shuffled_indices)

        with h5py.File(output_name, 'w') as output_file:
            # For each dataset in the file
            for key in file.keys():
                # Create a list to store shuffled data
                shuffled_data = []
                
                # Append shuffled trajectories to the list
                for idx in shuffled_indices:
                    shuffled_data.append(file[key][start_indices[idx]:end_indices[idx]+1])

                # Concatenate the shuffled data and save to the output file
                output_file.create_dataset(key, data=np.concatenate(shuffled_data))

def concatenate_and_shuffle(file_names, concatenated_name, shuffled_name):
    concatenate_hdf5(file_names, concatenated_name)
    shuffle_trajectories(concatenated_name, shuffled_name)

# Example of usage
file_names = ['/home/paperspace/decision-diffuser/code/analysis/newdataset/withoutGoal/my_datasetEnv3_rechtsrunter121123.hdf5',
               '/home/paperspace/decision-diffuser/code/analysis/newdataset/withoutGoal/my_datasetEnv3_linksrunter121123.hdf5'
              ]
concatenated_file = '/home/paperspace/decision-diffuser/code/analysis/newdataset/withoutGoal/env3-con.hdf5'
shuffled_file = '/home/paperspace/decision-diffuser/code/analysis/newdataset/withoutGoal/env33-shuffle.hdf5'
concatenate_and_shuffle(file_names, concatenated_file, shuffled_file)

# # Example of usage
# file_1 = '/home/paperspace/decision-diffuser/code/analysis/env2-m.hdf5'
# #file_1 = '/home/paperspace/decision-diffuser/code/analysis/env2-lefttotop-1.6k.hdf5'
# file_2 = '/home/paperspace/decision-diffuser/code/analysis/env2-lefttoright-1.7k.hdf5'
# output_file = '/home/paperspace/decision-diffuser/code/analysis/env2-concatonate.hdf5'
# final_file = '/home/paperspace/decision-diffuser/code/analysis/env2-lm.hdf5'
# concatenate_and_shuffle(file_1, file_2, output_file, final_file)

def remove_goal_observations(input_file, output_file):
    # Open the input file and the output file
    with h5py.File(input_file, 'r') as hf, h5py.File(output_file, 'w') as hf_out:
        # Copy all datasets except for 'observations' and 'next_observations' to the output file
        for ds in hf:
            if ds not in ['observations', 'next_observations']:
                hf.copy(ds, hf_out)
                
        # For 'observations' and 'next_observations', only keep the first two elements of each observation
        for ds in ['observations', 'next_observations']:
            data = hf[ds][:, :2]
            hf_out.create_dataset(ds, data=data)
        
    print(f"Modified dataset saved to: {output_file}")

# Example usage:
#input_file = '/home/paperspace/decision-diffuser/code/analysis/env3-comb5k.hdf5'
#output_file = '/home/paperspace/decision-diffuser/code/analysis/env3-comb-2obs-5k.hdf5'
#remove_goal_observations(input_file, output_file)







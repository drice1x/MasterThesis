import pandas as pd
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import ast

file_path = "/home/paperspace/decision-diffuser/code/analysis/latentdata.csv"
df = pd.read_csv(file_path)
#print(df.head(50))

# Pivot the DataFrame to reshape it
#pivoted_df = df.pivot( columns=['List_Type'], values='Tensor_Value', aggfunc=lambda x: x)

# Reset the index if you want 'Key' to be a column again
#pivoted_df.reset_index(inplace=True)

# Print the pivoted DataFrame
#print(pivoted_df)
# Group by 'Key' and 'List_Type', then aggregate the 'Tensor_Value' using a list
grouped = df.groupby(['Key', 'List_Type'])['Tensor_Value'].apply(list).reset_index()

# Pivot the grouped DataFrame to reshape it
pivoted_df = grouped.pivot(index='Key', columns='List_Type', values='Tensor_Value')

# Reset the index if you want a normal index
pivoted_df.reset_index(inplace=True)

# Print the pivoted DataFrame
#print(pivoted_df)

# Create a new DataFrame with 'Key' as columns
new_df = pd.DataFrame()

# Add 'Key' column
new_df['Key'] = pivoted_df['Key']

# Add 'action' column as is
new_df['action'] = pivoted_df['action']

# Add 'decoded' column as is
new_df['decoded'] = pivoted_df['decoded']

# Define the padding value
padding_value = 'PADDED'

# Expand 'latent' column into separate columns with padding
max_latent_length = pivoted_df['latent'].apply(len).max()
for i in range(max_latent_length):
    new_df[f'latent{i}'] = pivoted_df['latent'].apply(lambda x: x[i] if len(x) > i else padding_value)

# Print the new DataFrame
print(new_df)
print(new_df.columns)
print(type(new_df.latent0.loc[:0][0]))
print(new_df.latent0.loc[:0][0])
# Assuming you have a DataFrame called new_df and column names for latent0, latent1, ..., latent7
latent_columns = ['latent0', 'latent1', 'latent2', 'latent3', 'latent4', 'latent5', 'latent6']


########################################################
# Decoded KDE
########################################################
# Convert string representations of arrays to actual arrays for the 'decoded' column
new_df['decoded'] = new_df['decoded'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

# Initialize a dictionary to store KDE estimators for each time step
kde_estimators = {}

# Calculate KDE for each time step
for time_step in range(16):
    kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    
    # Preprocess the 'decoded' arrays for the current time step
    processed_arrays = new_df['decoded'].apply(lambda x: x[time_step])
    
    # Flatten the arrays and remove newline characters
    flattened_arrays = processed_arrays.apply(lambda x: [float(val) for val in x.split() if val != '\\n'])
    
    # Convert the flattened arrays to a NumPy array
    flattened_grid = np.array(flattened_arrays.tolist())
    
    # Reshape the flattened grid into (6, 6) matrices
    reshaped_grid = flattened_grid.reshape(len(flattened_arrays), 6, 6)
    
    # Fit the KDE model on the flattened grid
    kde.fit(flattened_grid)
    
    kde_estimators[time_step] = kde

# Create a time series DataFrame to visualize KDE over time
time_series_data = {
    'Time': range(16),
}

# Calculate KDE values for each time step
for time_step in range(16):
    kde = kde_estimators[time_step]
    
    # Calculate KDE scores for each position in the flattened grid
    kde_values = kde.score_samples(flattened_grid)
    time_series_data[f'TimeStep{time_step}'] = kde_values

time_series_df = pd.DataFrame(time_series_data)

# Create line plots for each time step
plt.figure(figsize=(12, 6))
for time_step in range(16):
    time_step_name = f'TimeStep{time_step}'
    plt.plot(time_series_df['Time'], time_series_df[time_step_name], label=time_step_name)

plt.xlabel('Time')
plt.ylabel('KDE Value')
plt.legend()
plt.title('KDE Over Time for Decoded Grids')
plt.show()

# Map the movement of the distribution to actions
for time_step in range(16):
    time_step_name = f'TimeStep{time_step}'
    kde = kde_estimators[time_step]
    
    # Calculate the action based on the maximum KDE value
    action = actions[np.argmax(kde_values)]
    print(f'Time Step {time_step}: {action}')
##############################################################
#############################################################
##############################################################
print("JA")







# Iterate through columns and convert the 'latent' columns to NumPy arrays
for column_name in new_df.columns:
    if column_name.startswith('latent'):
        new_df[column_name] = new_df[column_name].values.apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)


# Initialize a dictionary to store KDE estimators for each latent column
kde_estimators = {}

# Calculate KDE for each latent column
for i in range(8):
    column_name = f'latent{i}'
    kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde.fit(new_df[column_name].values)
    kde_estimators[column_name] = kde

# Create a time series DataFrame to visualize KDE over time
time_series_data = {
    'Time': range(16),
}

# Calculate KDE values for each latent column at each time step
for i in range(8):
    column_name = f'latent{i}'
    kde = kde_estimators[column_name]
    kde_values = kde.score_samples(list(new_df[column_name].values))
    time_series_data[column_name] = kde_values

time_series_df = pd.DataFrame(time_series_data)

# Create line plots for each latent variable
plt.figure(figsize=(12, 6))
for i in range(8):
    column_name = f'latent{i}'
    plt.plot(time_series_df['Time'], time_series_df[column_name], label=column_name)

plt.xlabel('Time')
plt.ylabel('KDE Value')
plt.legend()
plt.title('KDE Over Time for Latent Variables')
plt.show()
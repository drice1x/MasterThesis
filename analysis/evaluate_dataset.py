import h5py
import os
from diffuser.utils.arrays import to_torch, to_np, to_device

#file_size = os.path.getsize("/home/paperspace/decision-diffuser/code/analysis/my_datasetenv1-2obs-small.hdf5")
# print(file_size / 1000000)  # prints size of the file in bytes
'''
for i in range(16):
    action = i

    x_index = action // 4
    y_index = action % 4

    x_movement = [-1, -0.5, 0.5, 1][x_index]
    y_movement = [-1, -0.5, 0.5, 1][y_index]

'''
    #print(f"x_idx: {x_index}, y_idx: {y_index}, x_movement: {x_movement}, y_movement: {y_movement}")

def determine_action(state, new_state):
    x_diff = new_state[0] - state[0]
    y_diff = new_state[1] - state[1]

    # Available movement options
    movements = [-1, -0.5, 0.5, 1]

    # Finding the closest movement values from available options
    x_movement = closest_value(x_diff, movements)
    y_movement = closest_value(y_diff, movements)

    x_mappings = {-1: 0, -0.5: 1, 0.5: 2, 1: 3}
    y_mappings = {-1: 0, -0.5: 1, 0.5: 2, 1: 3}

    x_index = x_mappings[x_movement]
    y_index = y_mappings[y_movement]

    action = x_index * 4 + y_index
    return action

'''for i in range(16):
    x_index = i // 4  
    y_index = i % 4
    x_movement = [-1, -0.5, 0.5, 1][x_index]
    y_movement = [-1, -0.5, 0.5, 1][y_index]
    # print(f"action: {i} | x_movement: {x_movement} | y_movement{y_movement}")'''

def closest_value(val, options):
    return min(options, key=lambda x: abs(x - val))


    state = [-0.6, 0.985]
    new_state = [-1.15, 0.995]

    # print(f"action was {determine_action(state, new_state)}")
    # print(f"action was {action}")


def print_structure(file_name):
    with h5py.File(file_name, 'r') as file:
        file.visit(print)


# Example usage:
# print_structure('/home/paperspace/decision-diffuser/code/analysis/maze2d-medium-dense-v1.hdf5')
# print("Our File")
# ‚print_structure('/home/paperspace/decision-diffuser/code/analysis/my_datasetenv1_2obs.hdf5')

def display_first_n_values(filename, n=5):
    with h5py.File(filename, 'r') as file:

        def explore_element(name, obj):
            # Ensure that the data is a dataset, has a shape, and is not empty
            if isinstance(obj, h5py.Dataset) and obj.shape and obj.shape[0] > 0:
                if name == "rewards":
                    print(f"{name}: {obj[:n]}")

        # Traverse all datasets regardless of their depth in the hierarchy
        file.visititems(explore_element)


# Usage
# display_first_n_values('/home/paperspace/decision-diffuser/code/analysis/maze2d-medium-dense-v1.hdf5', n=1000)
# display_first_n_values('/home/paperspace/decision-diffuser/code/analysis/my_datasetenv1_2obs_normr.hdf5', n=100)


import h5py


def analyze_trajectories(file_name):
    with h5py.File(file_name, 'r') as file:
        rewards = file['rewards'][:]  # Assuming the rewards are stored in a dataset named "rewards"

        goal_found_count = 0
        collision_free_count = 0
        step_count = 0
        collision_in_current_trajectory = False

        for reward in rewards:
            if reward == 20:
                goal_found_count += 1
                if not collision_in_current_trajectory:
                    collision_free_count += 1
                collision_in_current_trajectory = False
                step_count = 0  # Reset step count after goal is found
            elif reward == -15:
                collision_in_current_trajectory = True

            step_count += 1

            # Check for failed trajectory
            if step_count == 15:
                if not collision_in_current_trajectory:
                    collision_free_count += 1
                collision_in_current_trajectory = False
                step_count = 0  # Reset step count after 15 steps

        return goal_found_count, collision_free_count


#goal_count, no_collision_count = analyze_trajectories(
#   '/home/paperspace/decision-diffuser/code/analysis/my_datasetenv1-2obs-small.hdf5')
# print(f"Number of trajectories that found the goal: {goal_count}")
# print(f"Number of collision-free trajectories: {no_collision_count}")

import h5py
import numpy as np


def analyze_and_print_trajectories(file_name):
    with h5py.File(file_name, 'r') as file:
        rewards = file['rewards'][:]
        observations = file['observations'][:]

        goal_found_count = 0
        collision_free_count = 0
        collision_then_goal_count = 0
        step_count = 0
        collision_in_current_trajectory = False
        trajectory_start_index = 0

        total_length = 0
        goal_lengths = []
        collision_lengths = []
        total_lengths = []

        example_no_goal_no_collision = None
        example_collision_then_goal = None

        for idx, reward in enumerate(rewards):
            if reward == 1:
                goal_found_count += 1
                total_lengths.append(step_count + 1)
                if collision_in_current_trajectory:
                    collision_then_goal_count += 1
                    if example_collision_then_goal is None:
                        example_collision_then_goal = observations[trajectory_start_index:idx + 1]
                elif not collision_in_current_trajectory:
                    collision_free_count += 1
                elif not collision_in_current_trajectory and example_no_goal_no_collision is None:
                    example_no_goal_no_collision = observations[trajectory_start_index:idx + 1]
                goal_lengths.append(step_count + 1)
                collision_in_current_trajectory = False
                step_count = 0
                trajectory_start_index = idx + 1
            elif reward == -0.75:
                collision_in_current_trajectory = True
                collision_lengths.append(step_count + 1)

            step_count += 1

            if step_count == 15:
                total_lengths.append(step_count)
                if not collision_in_current_trajectory:
                    collision_free_count += 1
                    if example_no_goal_no_collision is None:
                        example_no_goal_no_collision = observations[trajectory_start_index:idx + 1]
                collision_in_current_trajectory = False
                step_count = 0
                trajectory_start_index = idx + 1

        average_length = np.mean(total_lengths)
        average_goal_length = np.mean(goal_lengths)
        average_collision_length = np.mean(collision_lengths) if collision_lengths else 0

        # print("Examples:")
        # print(f"Trajectory without goal and collision: {example_no_goal_no_collision}")
        # print(f"Trajectory with collision then found goal: {example_collision_then_goal}")
        # print(f"Avg length of all trajectories: {average_length}")
        # print(f"Avg length of trajectories that found the goal: {average_goal_length}")
        # print(f"Avg length of trajectories that have collisions: {average_collision_length}")

        return goal_found_count, collision_free_count


##goal_count, no_collision_count = analyze_and_print_trajectories(
#    '/home/paperspace/decision-diffuser/code/analysis/my_datasetenv1-2obs-small.hdf5')
# print(f"Number of trajectories that found the goal: {goal_count}")
# print(f"Number of collision-free trajectories: {no_collision_count}")

def get_actions(trajectory, goal_idx, normalize=True):
   
    if normalize:
        trajectory = trajectory*12
    
    trajectory = trajectory[:goal_idx+1]

    movements = [(i, j) for i in [-1, -0.5, 0.5, 1] for j in [-1, -0.5, 0.5, 1]]

    def find_confident_transition(start_idx, trajectory):
        """Finds the next 'confident' transition and returns the end index."""
        for i in range(start_idx, len(trajectory) - 1):
            delta = trajectory[i+1] - trajectory[i]
            for movement in movements:
                if np.isclose(delta, movement, atol=0.1).all():
                    return i+1
        return None  

    def possible_actions_for_delta(delta):
        """Returns possible action sequences for a given delta."""
        results = []
        for i, m1 in enumerate(movements):
            for j, m2 in enumerate(movements):
                combined_movement = np.add(m1, m2)
                if np.isclose(delta, combined_movement, atol=0.24).all():
                    results.append((i, j))
        if results == []:
            results.append((4, 8))
        return results

    actions_taken = []

    i = 0
    while i < len(trajectory) - 1:
        s1 = trajectory[i]
        next_confident_idx = find_confident_transition(i, trajectory)
        
        if next_confident_idx is None:
            # if there's no confident transition ahead, just choose the closest action for the current delta
            delta = trajectory[i+1] - s1
            distances = [np.linalg.norm(delta - np.array(movement)) for movement in movements]
            closest_movement_idx = np.argmin(distances)
            actions_taken.append(closest_movement_idx)
            i += 1
            continue
        
        s2 = trajectory[next_confident_idx]
        delta = s2 - s1
        
        # Find actions that can explain the transition from s1 to s2
        possible_actions = possible_actions_for_delta(delta)
        # Here, just taking the first set of actions, but other logic can be used
        actions_taken.extend(possible_actions[0])
        
        i = next_confident_idx  # jump to the next confident state

    return actions_taken


def estimate_actions(trajectory, goal_idx=None, normalize=False, noise_factor=1e-6):
    if normalize:
        trajectory = trajectory * 20

    if goal_idx is None:
        previous_array = None

        # Iterate through the entries
        for i in range(len(trajectory)):
            current_array = trajectory[i]
            
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
            index_of_last_change = len(trajectory) - 1
        trajectory = trajectory[:index_of_last_change + 1]
    else:
        trajectory = trajectory[:goal_idx + 1]

    x_transitions = trajectory[:, 0]
    y_transitions = trajectory[:, 1]
    
    def _continuous_to_discrete_action(self, continuous_action):
            # Map the continuous action values to discrete directions
            angle = np.arctan2(continuous_action[1], continuous_action[0])
            angle_deg = np.degrees(angle) % 360

            if 22.5 <= angle_deg < 67.5:      # North-East
                return 5
            elif 67.5 <= angle_deg < 112.5:   # North
                return 0
            elif 112.5 <= angle_deg < 157.5:  # North-West
                return 4
            elif 157.5 <= angle_deg < 202.5:  # West
                return 2
            elif 202.5 <= angle_deg < 247.5:  # South-West
                return 6
            elif 247.5 <= angle_deg < 292.5:  # South
                return 1
            elif 292.5 <= angle_deg < 337.5:  # South-East
                return 7
            else:                            # East
                return 3


        # Combine x and y movements into continuous action tuples
    continuous_actions = np.vstack((x_transitions, y_transitions)).T

    # Convert continuous actions to discrete actions
    discrete_actions = [_continuous_to_discrete_action(action) for action in continuous_actions]

    print(discrete_actions)

    #x_movements = np.diff(x_transitions.cpu())
    #y_movements = np.diff(y_transitions.cpu())
    
    # Find zero movements and add noise
    #x_zero_indices = np.where(x_movements == 0)[0]
    #y_zero_indices = np.where(y_movements == 0)[0]

    #x_movements[x_zero_indices] += np.random.uniform(-noise_factor, noise_factor, x_zero_indices.shape)
    #y_movements[y_zero_indices] += np.random.uniform(-noise_factor, noise_factor, y_zero_indices.shape)
        
        
    # Add noise to zero movements
    #x_noise = np.random.uniform(-noise_factor, noise_factor, x_movements.shape)
    #y_noise = np.random.uniform(-noise_factor, noise_factor, y_movements.shape)
    #x_movements[x_movements == 0] += x_noise[x_movements == 0]
    #y_movements[y_movements == 0] += y_noise[y_movements == 0]


    def get_discrete_action(continuous_action):
        #angles = np.arctan2(continuous_action[1], continuous_action[0])
        #angle_deg = np.degrees(angles) % 360
            # Calculate angle avoiding division by zero
        if x_movement != 0:
            angle = np.arctan(y_movement / x_movement)
        elif y_movement > 0:
            angle = np.pi / 2
        elif y_movement < 0:
            angle = -np.pi / 2
        else:
            angle = 0  # Both x and y movements are zero
        
        angle_deg = (np.degrees(angle) + 360) % 360  # Ensure positive angle

        action_mapping = {
            (22.5, 67.5): 5,
            (67.5, 112.5): 0,
            (112.5, 157.5): 4,
            (157.5, 202.5): 2,
            (202.5, 247.5): 6,
            (247.5, 292.5): 1,
            (292.5, 337.5): 7,
            (337.5, 22.5): 3,
        }

        for angle_range, action in action_mapping.items():
            if angle_range[0] <= angle_deg < angle_range[1]:
                return action

    '''    actions = []
        for x_movement, y_movement in zip(x_movements, y_movements):
            continuous_action = np.array([x_movement, y_movement])
            discrete_action = get_discrete_action(continuous_action)
            actions.append(discrete_action)'''

    #return actions

    def get_discrete_action1(continuous_action):
        x_movement, y_movement = continuous_action
        
        # Check for zero movements and assign a default action
        if np.all(np.isclose(continuous_action, 0, atol=1e-6)):
            return 8  # Assign a unique value to represent no movement
        
        angle = np.arctan2(y_movement, x_movement)
        angle_deg = (np.degrees(angle) + 360) % 360  # Ensure positive angle
        
        action_mapping = {
            (22.5, 67.5): 5,
            (67.5, 112.5): 0,
            (112.5, 157.5): 4,
            (157.5, 202.5): 2,
            (202.5, 247.5): 6,
            (247.5, 292.5): 1,
            (292.5, 337.5): 7,
            (337.5, 22.5): 3,
        }
        
        closest_distance = float('inf')
        closest_action = None
        
        if x_movement == 0 and y_movement == 0:
            return 8  # Return 8 if both movements are zero
        
        for angle_range, action in action_mapping.items():
            lower_bound, upper_bound = angle_range
            if lower_bound <= angle_deg < upper_bound:
                return action
        
            # Calculate the distance to the middle of the angle range
            mid_angle = (lower_bound + upper_bound) / 2
            distance = min(abs(angle_deg - lower_bound), abs(angle_deg - upper_bound))
            
            if distance < closest_distance:
                closest_distance = distance
                closest_action = action
        
        return closest_action
    actions = [get_discrete_action1((x, y)) for x, y in zip(x_movements, y_movements)]
    return actions
    # Provided movements
    '''x_movements = np.array([ 1.8282701 ,  0.0036248 , -0.43958795, -0.11402875, -1.0479846 ,
            0.78245586, -0.0920738 ,  1.0793204 ,  0.        , -0.23230332,
        -0.46484536, -1.3028474 ], dtype=np.float32)

    y_movements = np.array([ 1.8447292 , -1.1149031 , -0.19403198, -0.38834596,  1.1311613 ,
            0.37498248, -1.3938193 , -0.25976884,  0.93587196,  0.25938937,
        -0.9148594 , -0.28040582], dtype=np.float32)'''

# Calculate actions
    
    
def extract_discrete_action_from_observations(obs_t, obs_t_plus_1, num_cols):
    # Reshape flattened observations back to 8x8 grid
    obs_t_grid = obs_t.reshape((4,4 ))
    obs_t_plus_1_grid = obs_t_plus_1.reshape((4, 4))
    
    # Find the positions of the agent (marked as "5")
    row_t, col_t = np.where(obs_t_grid == 30)
    row_t_plus_1, col_t_plus_1 = np.where(obs_t_plus_1_grid == 30)
    
    delta_row = row_t_plus_1[0] - row_t[0]
    delta_col = col_t_plus_1[0] - col_t[0]
    
    if delta_row == -1 and delta_col == 0:  # Up
        return 0 #"Up"
    elif delta_row == 1 and delta_col == 0:  # Down
        return 1 # "Down"
    elif delta_row == 0 and delta_col == -1:  # Left
        return 2 #"Left"
    elif delta_row == 0 and delta_col == 1:  # Right
        return 3 # "Right"
    elif delta_row == -1 and delta_col == -1:  # North-West
        return 4 #"North-West"
    elif delta_row == -1 and delta_col == 1:  # North-East
        return 5 #"North-East"
    elif delta_row == 1 and delta_col == -1:  # South-West
        return 6 #"South-West"
    elif delta_row == 1 and delta_col == 1:  # South-East
        return 7 #"South-East"
    else:
        return None  # No valid discrete action found

def normalize_observation( observation, observation_values = [0, 30]):
    min_value = min(observation_values)
    max_value = max(observation_values)
    normalized_observation = (observation - min_value) / (max_value - min_value) * 2 - 1
    return normalized_observation

def denormalize_observation( normalized_observation, observation_values = [0, 30]):
    min_value = min(observation_values)
    max_value = max(observation_values)
    denormalized_observation = (normalized_observation + 1) / 2 * (max_value - min_value) + min_value
    return denormalized_observation

def estimate_actions1(trajectory, goal_idx=None, normalize=False, noise_factor=1e-6):
    if normalize:
        trajectory = trajectory * 20

    if goal_idx is None:
        previous_array = None

        # Iterate through the entries
        for i in range(len(trajectory)):
            current_array = trajectory[i]
            
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
            index_of_last_change = len(trajectory) - 1
        trajectory = trajectory[:index_of_last_change + 1]
    else:
        trajectory = trajectory[:goal_idx + 1]

    denormtraj = denormalize_observation(trajectory)


  

def round_values2(arr):
    # Find positive values
    arr = to_np(arr)
    # Find the index of the maximum value
    max_index = np.argmax(arr) 
    arr[arr == np.max(arr)] = 1
    arr[arr < np.max(arr)] = -1
    return arr

def round_values1(arr):
    # Find positive values
    arr = to_np(arr)
    positive_values = arr[arr > 0]

    if len(positive_values) == 0:
        # If there are no positive values, round the highest value to 1
        arr[arr == np.max(arr)] = 1
        # Find the second highest value and round it to 0
        unique_values = np.unique(arr)
        if len(unique_values) >= 2:
            second_highest_value = unique_values[-2]
            #arr[arr == second_highest_value] = 0
    elif len(positive_values) >= 2:
        # Sort positive values in descending order
        sorted_positive_values = np.sort(positive_values)[::-1]

        # Round the highest positive value to 1
        arr[arr == sorted_positive_values[0]] = 1

        # Round the second highest positive value to 0
        #arr[arr == sorted_positive_values[1]] = 0
    else:
        # If there's only one positive value, round it to 1
        arr[arr > 0] = 1

    # Round the remaining non-positive values to -1 or 0
    arr[arr < 0] = -1

    return arr

    
    
# Function to round values as specified
def round_values(arr):
    # Find positive values
    #arr = arr.cpu().numpy()
    arr = to_np(arr)
    
    #positive_values = [np.array(i)[i > 0]for i in arr]
    
    positive_values = arr[arr > 0]
    
    #values = np.array(positive_values)

    

    if len(positive_values) >= 2:
        # Sort positive values in descending order
        sorted_positive_values = np.sort(positive_values)[::-1]

        # Round the highest positive value to 1
        arr[arr == sorted_positive_values[0]] = 1

        # Round the second highest positive value to 0
        arr[arr == sorted_positive_values[1]] = 0
    else:
        # If there's only one positive value, round it to 1
        arr[arr > 0] = 1

    # Round the remaining non-positive values to -1 or 0
    arr[arr < 0] = -1

    return arr


# Function to find the position of "1" in the array
def find_position_with_value(array, value):
        indices = np.argwhere(array == value)
        if len(indices) > 0:
            return tuple(indices[0])
        else:
            return None



# Define the actions
actions = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "North-West",
    5: "North-East",
    6: "South-West",
    7: "South-East"
}

# Function to calculate the action that led to the position change
def calculate_action1(position1, position2):
        # Find the position of "1" in array1 and array2
    position1 = find_position_with_value(position1.reshape((6,6)), 1)
    position2 = find_position_with_value(position2.reshape((6,6)), 1)
    if position1 is None or position2 is None:
        return None, "One or both positions not found"

    row1, col1 = position1
    row2, col2 = position2

    # Calculate the action based on the change in position
    if row2 < row1:
        if col2 < col1:
            action = 4  # North-West
        elif col2 > col1:
            action = 5  # North-East
        else:
            action = 0  # Up
    elif row2 > row1:
        if col2 < col1:
            action = 6  # South-West
        elif col2 > col1:
            action = 7  # South-East
        else:
            action = 1  # Down
    else:
        if col2 < col1:
            action = 2  # Left
        elif col2 > col1:
            action = 3  # Right
        else:
            action = None  # No change in position

    return action

def calculate_action(position1, position2):
    # Find the position of "1" in array1 and array2
    position1 = find_position_with_value(position1.reshape((4,4)), 1)
    position2 = find_position_with_value(position2.reshape((4,4)), 1)
    if position1 is None or position2 is None:
        return None, "One or both positions not found"

    row1, col1 = position1
    row2, col2 = position2

    # Calculate the action based on the change in position
    if row2 < row1:
        action = 0  # North
    elif row2 > row1:
        action = 1  # South
    elif col2 < col1:
        action = 2  # West
    elif col2 > col1:
        action = 3  # East
    else:
        action = None  # No change in position

    return action

# Define the gradient calculation function
def calculate_gradient(matrix, row, col):
        gradient_x = np.gradient(matrix, axis=0)
        gradient_y = np.gradient(matrix, axis=1)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        return gradient_magnitude[row, col], gradient_direction[row, col]
    
# Define the gradient calculation function
def calculate_gradient_between_arrays(array1, array2, row, col):
    gradient_x = np.gradient(array1 - array2, axis=0)
    gradient_y = np.gradient(array1 - array2, axis=1)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude[row, col], gradient_direction[row, col]

def gradient_action(agentposition, position1, position2):

    # Define the actions
    actions = {
        0: "Up",
        1: "Down",
        2: "Left",
        3: "Right",
        4: "North-West",
        5: "North-East",
        6: "South-West",
        7: "South-East"
    }

    # Find the starting position (highest value)
    #starting_position = np.unravel_index(np.argmax(position1), position1.shape)
    row, col = agentposition

    # Calculate gradient at the starting position
    #gradient_magnitude, gradient_direction = calculate_gradient(matrix=position1, row=row, col=col)
    
    #Calculate gradient between array1 and array2 at the agent's position
    gradient_magnitude, gradient_direction = calculate_gradient_between_arrays(position1.cpu().detach().numpy(), position2.cpu().detach().numpy(), row, col)


    # Determine the action based on gradient direction
    if -np.pi/8 <= gradient_direction < np.pi/8:
        action = 3#actions[3]  # Right
    elif np.pi/8 <= gradient_direction < 3*np.pi/8:
        action = 7#actions[7]  # South-East
    elif 3*np.pi/8 <= gradient_direction < 5*np.pi/8:
        action = 1#actions[1]  # Down
    elif 5*np.pi/8 <= gradient_direction < 7*np.pi/8:
        action = 6#actions[6]  # South-West
    elif abs(gradient_direction) >= 7*np.pi/8:
        action = 2#actions[2]  # Left
    elif -7*np.pi/8 <= gradient_direction < -5*np.pi/8:
        action = 4#actions[4]  # North-West
    elif -5*np.pi/8 <= gradient_direction < -3*np.pi/8:
        action = 0#actions[0]  # Up
    else:
        action = 5#actions[5]  # North-East
        
    return action

    #print(f"Agent's action at starting position ({row}, {col}): {action} (Gradient Magnitude: {gradient_magnitude})")
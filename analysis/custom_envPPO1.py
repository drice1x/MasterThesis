import h5py
import gymnasium as gymm
from gymnasium import spaces
import numpy as np


###################################
###################################

# EXPERIMENT 1

###################################
###################################

gridsize = {
    0: (4, 4),
    1: (6, 6),
    2: (8, 8)
}


maxsteps = {
    0: 8,
    1: 8,
    2: 8
}

envselector = 0 #np.random.randint(0, 3)# choose which grid is displayed

class CustomEnv1(gymm.Env):
    def __init__(self,grid_size = gridsize[envselector], max_steps=  maxsteps[envselector]):
        super().__init__()

            
        self.grid_size = grid_size
        self.num_rows, self.num_cols = grid_size[0], grid_size[1]

        #self.observation_space = spaces.Box(low=-1., high=1., shape=(self.num_rows * self.num_cols,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1., high= 1., shape=(self.num_rows * self.num_cols,), dtype=np.float32)
        #self.action_space = spaces.Box(low=-1., high=1.,shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        self.starting_pos = (np.random.randint(0, 3), np.random.randint(0, 3)) #completely random start position
        self.agent_position = self.starting_pos
        self.goal_position = (np.random.randint(self.num_rows-2, self.num_rows), np.random.randint(self.num_rows-2, self.num_rows)) #(np.random.randint(self.num_rows-1, self.num_rows-1), np.random.randint(self.num_rows-1, self.num_rows-1))
        self.max_steps = max_steps
        self.current_step = 0
        self.visited_positions = []

    def reset(self, *, seed=None, options=None):
        
        

        # Create the initial observation
        #observation = self._get_observation()
        #normalized_observation = self.normalize_observation(observation)
        #self.obstacle_positions = self._generate_obstacle_positions()
        #possiblestarts = [(np.random.randint(0, 2), np.random.randint(0, 1)), (np.random.randint(2, 4), np.random.randint(0, 1))]

        possiblestarts = [(0,0), (0,3)]  #[(0,0)] 
        possibleends = [(3,3), (3,0)] #[(3,3)] 

        randsel = 0#np.random.randint(0,2)
        #self.agent_position , self.goal_position
        self.agent_position, self.goal_position = possiblestarts[randsel],possibleends[randsel]#possiblestarts[randsel], possibleends[randsel]
        self.starting_pos = self.agent_position
        self.visited_positions = []
        self.current_step = 0
        infodict = {}
        return self._get_observation(), infodict
    
    def target(self):
        observation1 = np.zeros((self.num_rows, self.num_cols), dtype=np.int8)
        observation1[self.goal_position] = 1
        #for obstacle_position, obstacle_size in self.obstacle_positions:
        #    for i in range(obstacle_size[0]):
        #        for j in range(obstacle_size[1]):
        #            observation1[obstacle_position[0] + i, obstacle_position[1] + j] = 20
        #normalize_observation(observation1)
        return self.normalize_observation(observation1.flatten()) #self.normalize_observation(observation1.flatten()) #self.goal_position
    
    def normalize_observation(self, observation):
        min_value = 0
        max_value = 1
        #(observation-np.min(observation))/(np.max(observation)-np.min(observation))
        normalized_observation = ((observation - min_value) / (max_value - min_value))
        return normalized_observation

    def denormalize_observation(self, normalized_observation):
        min_value = 0
        max_value = 30
        denormalized_observation = (normalized_observation + 1) / 2 * (max_value - min_value) + min_value
        return denormalized_observation
        
    def _get_observation(self):
        observation = np.zeros((self.num_rows, self.num_cols), dtype=np.int8)
        # Mark visited positions as 2
        #for position in self.visited_positions:
        #    observation[position] = 2

        # Mark obstacle positions as 20
        #for obstacle_position, obstacle_size in self.obstacle_positions:
        #    for i in range(obstacle_size[0]):
        #        for j in range(obstacle_size[1]):
        #            observation[obstacle_position[0] + i, obstacle_position[1] + j] = 20

        # Mark the agent's current position as 30
        observation[self.agent_position] = 1
        # Mark goal position as 10
        observation[self.goal_position] = 0.5
        
        #normalized_observation = self.normalize_observation(observation.flatten())

        #return normalized_observation
        return  self.normalize_observation(observation.flatten())#observation.flatten()#self.normalize_observation(observation.flatten()) #self.normalize_observation(observation.flatten())#np.array(self.agent_position, dtype=np.int8) #observation.flatten() #np.array(self.agent_position, dtype=np.int8) #observation.flatten()#self.agent_position#observation.flatten()
    
    def _continuous_to_discrete_action(self, continuous_action):

        #if np.all(continuous_action == 0):
        #    return 8 
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

    def _calculate_reward1(self, new_position):
        goal_row, goal_col = self.goal_position
        current_row, current_col = new_position

        if new_position == self.goal_position:
            return 1  # Positive reward for reaching the goal

        if self.current_step > self.max_steps:
            return -1  # Negative reward if current step exceeds max_steps

        if self._is_collision(new_position):
            return -1/3  # Negative reward for collisions

        # Calculate the Euclidean distance to the goal
        distance_to_goal = ((goal_row - current_row) ** 2 + (goal_col - current_col) ** 2) ** 0.5

        # Normalize the distance-based reward between -1 and 1
        d_min = 0  # Minimum possible distance
        d_max = ((self.grid_size[0] - 1) ** 2 + (self.grid_size[1] - 1) ** 2) ** 0.5  # Maximum possible distance
        scaled_distance = 2 * (distance_to_goal - d_min) / (d_max - d_min) - 1

        return -1 #distance_to_goal
    
    def _calculate_reward(self, new_position):
        goal_row, goal_col = self.goal_position
        current_row, current_col = new_position

        if new_position == self.goal_position:
            return 30  # Positive reward for reaching the goal

        if self.current_step > self.max_steps:
            return -30  # Negative reward if current step exceeds max_steps

        if self._is_collision(new_position):
            return -10  # Negative reward for collisions

        # Calculate the Euclidean distance to the goal
        distance_to_goal = ((goal_row - current_row) ** 2 + (goal_col - current_col) ** 2) ** 0.5

        # Normalize the distance-based reward between -1 and 1
        #d_min = 0  # Minimum possible distance
        #d_max = ((self.grid_size[0] - 1) ** 2 + (self.grid_size[1] - 1) ** 2) ** 0.5  # Maximum possible distance
        #scaled_distance = 2 * (distance_to_goal - d_min) / (d_max - d_min) - 1

        return -1
    
    
    def step(self, action):
        done = False
        truncated = False
        row, col = self.agent_position
        new_row, new_col = row, col

        
        #if not isinstance(action, (np.int64, int)):
        #    action = self._continuous_to_discrete_action(action)

        #if action == 8:  # No movement
        #    pass  # Agent remains in the current position
        if action == 0:  # Up
            new_row = max(row - 1, 0)
        elif action == 1:  # Down
            new_row = min(row + 1, self.num_rows - 1)
        elif action == 2:  # Left
            new_col = max(col - 1, 0)
        elif action == 3:  # Right
            new_col = min(col + 1, self.num_cols - 1)
        elif action == 4:  # North-West
            new_row = max(row - 1, 0)
            new_col = max(col - 1, 0)
        elif action == 5:  # North-East
            new_row = max(row - 1, 0)
            new_col = min(col + 1, self.num_cols - 1)
        elif action == 6:  # South-West
            new_row = min(row + 1, self.num_rows - 1)
            new_col = max(col - 1, 0)
        elif action == 7:  # South-East
            new_row = min(row + 1, self.num_rows - 1)
            new_col = min(col + 1, self.num_cols - 1)

        new_position = (new_row, new_col)

        # Check if the new position is valid
        if not self._is_collision(new_position):
            self.agent_position = new_position

        self.current_step +=1
        # Calculate the reward based on the new position
        reward = self._calculate_reward(new_position)
        
        if reward == 30:  # Positive reward for reaching the goal
            done = True
        #elif reward == -10:  # Negative reward for colliding with an obstacle
        #    done = True
        if self.current_step >= self.max_steps:
            truncated=True
            done = True
        #if truncated and self.agent_position != self.goal_position:
        #    reward = -1  # Negative reward for reaching the maximum steps without reaching the goal

        return self._get_observation(), reward, done, truncated, {}
    #(obs, rew, terminated, truncated, info
    # (obs, rew, done, info)
    
    

    #def get_dataset(self):
    #    return self.dataset
    
    def _is_collision(self, position):
        #Check if Agent hits obstacle
        #for obstacle_position, obstacle_size in self.obstacle_positions:
        #    if position[0] in range(obstacle_position[0], obstacle_position[0] + obstacle_size[0]) and \
        #            position[1] in range(obstacle_position[1], obstacle_position[1] + obstacle_size[1]):
        #        return True

        # Check if the agent hits the wall
        if position[0] < 0 or position[0] >= self.num_rows or \
                position[1] < 0 or position[1] >= self.num_cols:
            return True

        return False

    def close(self):
        self.dataset.close()
    
    def is_position_valid (self, position, obstacle_positions):
        for obstacle_pos, obstacle_size in obstacle_positions:
            obs_row_start, obs_col_start = obstacle_pos
            obs_row_end = obs_row_start + obstacle_size[0]
            obs_col_end = obs_col_start + obstacle_size[1]
            
            row, col = position
            if obs_row_start <= row < obs_row_end and obs_col_start <= col < obs_col_end:
                return False
        return True
    
    
    def generate_valid_start_goal(self, obstacle_positions):
        for _ in range(50):
            start_position = (np.random.randint(0, 2), np.random.randint(0, 2))
            goal_position = (np.random.randint(self.num_rows - 2, self.num_rows), np.random.randint(self.num_rows - 2, self.num_rows))
            
            if self.is_position_valid(start_position, obstacle_positions) and self.is_position_valid(goal_position, obstacle_positions):
                return start_position, goal_position
                break
            
            
            
    def _generate_obstacle_positions(self):
        #generating obstacles
        #obstacle_coordinates = [(2, 3), (6, 3), (4, 5)]
        obstacle_coordinates = [(3,3)]
        num_obstacles = len(obstacle_coordinates)
        obstacle_positions = []
        #num_obstacles = 3 #np.random.randint(1, 3)  # Randomly select the number of obstacles
        for _ in range(num_obstacles):
            obstacle_row = 3
            obstacle_col = 3
            obstacle_size = (1, 1)
            #obstacle_size = (np.random.randint(1, 2), np.random.randint(1, 2))
            obstacle_positions.append(((obstacle_coordinates[_]), obstacle_size))
            #obstacle_row = np.random.randint(2, 5)  # Random row position
            #obstacle_col = np.random.randint(2, 5)  # Random column position
            #obstacle_size = (np.random.randint(1, 3), np.random.randint(1, 3))  # Random obstacle size
            #obstacle_positions.append(((obstacle_row, obstacle_col), obstacle_size))
        return obstacle_positions
    
    @property
    def _max_episode_steps(self):
        return self.max_episode_steps
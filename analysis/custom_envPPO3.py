import h5py
import gymnasium as gymm
from gymnasium import spaces
import numpy as np
import math
#import gymnasium as gym
#from gymnasium import spaces

###################################
###################################

# EXPERIMENT 3

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
envselector = 2 #np.random.randint(0, 3)# choose which grid is displayed
goalselector = np.random.randint(0,2)

class CustomEnv3(gymm.Env):
    def __init__(self,grid_size = gridsize[envselector], max_steps=  maxsteps[envselector], goalselector = goalselector):
        super().__init__()
            
        self.goalselector = goalselector
        self.grid_size = grid_size
        self.num_rows, self.num_cols = grid_size[0], grid_size[1]

        #self.observation_space = spaces.Box(low=-1., high=1., shape=(self.num_rows * self.num_cols,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1., high= 1., shape=(self.num_rows * self.num_cols,), dtype=np.float32)
        #self.action_space = spaces.Box(low=-1, high=1,shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        self.starting_pos = (1,1) #completely random start position
        self.agent_position = self.starting_pos
        self.goal_position = (3,3) #(np.random.randint(self.num_rows-1, self.num_rows-1), np.random.randint(self.num_rows-1, self.num_rows-1))
        self.max_steps = max_steps
        self.current_step = 0
        self.visited_positions = []

    def reset(self, *, seed=None, options=None):
        
        self.goalselector = self.goalselector
        self.obstacle_positions = [(6,1), (6,6)]#self._generate_obstacle_positions()
        self.agent_position, self.goal_position = self.generate_valid_start_goal()
        self.starting_pos = self.agent_position
        self.visited_positions = []
        self.current_step = 0
        
        infodict = {}
        return self._get_observation(), infodict
    
    
    
    def target(self):
        observation1 = np.zeros((self.num_rows, self.num_cols), dtype=np.int8)
        observation1[self.goal_position] = 1
        #for obstacle_position in self.obstacle_positions:
        #    observation1[obstacle_position] = 20
        return self.normalize_observation(observation1).flatten()#self.normalize_observation(observation1.flatten()) #self.goal_position
    
    
    def normalize_observation(self, observation):
        min_value = 0
        max_value = 1  # Change the max_value to 1 for 0-1 normalization
        normalized_observation = (observation - min_value) / (max_value - min_value)
        return normalized_observation
    def denormalize_observation(self, normalized_observation):
        min_value = 0
        max_value = 30
        denormalized_observation = (normalized_observation + 1) / 2 * (max_value - min_value) + min_value
        return denormalized_observation
        
    def _get_observation(self):
        observation = np.zeros((self.num_rows, self.num_cols), dtype=np.int8)
        # Mark the agent's current position as 30
        observation[self.agent_position] = 1
        # Mark obstacle position as 20
        #for obstacle_position in self.obstacle_positions:
        #    observation[obstacle_position] = 20
        #observation[self.obstacle_positions] = 20

        observation[self.goal_position] = 0.5
        #return normalized_observation
        return self.normalize_observation(observation).flatten()#self.normalize_observation(observation.flatten())#np.array(self.agent_position, dtype=np.int8) #observation.flatten() #np.array(self.agent_position, dtype=np.int8) #observation.flatten()#self.agent_position#observation.flatten()
    
    def _continuous_to_discrete_action(self, continuous_action):

        if np.all(continuous_action == 0):
            return 8 
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
            return 100  # Positive reward for reaching the goal

        if self.current_step > self.max_steps:
            return -100 # Negative reward if current step exceeds max_steps

        if self._is_collision(new_position):
            return -100/3  # Negative reward for collisions

        # Calculate the Euclidean distance to the goal
        distance_to_goal = ((goal_row - current_row) ** 2 + (goal_col - current_col) ** 2) ** 0.5

        # Normalize the distance-based reward between -1 and 1
        d_min = 0  # Minimum possible distance
        d_max = ((self.grid_size[0] - 1) ** 2 + (self.grid_size[1] - 1) ** 2) ** 0.5  # Maximum possible distance
        scaled_distance = 2 * (distance_to_goal - d_min) / (d_max - d_min) - 1

        return -int(distance_to_goal)
    
    def _calculate_reward(self, new_position):
        goal_row, goal_col = self.goal_position
        current_row, current_col = new_position

        if new_position == self.goal_position:
            return 30  # Positive reward for reaching the goal

        if self.current_step > self.max_steps:
            return -30 # Negative reward if current step exceeds max_steps

        if self._is_collision(new_position):
            return -10  # Negative reward for collisions

        # Calculate the Euclidean distance to the goal
        distance_to_goal = ((goal_row - current_row) ** 2 + (goal_col - current_col) ** 2) ** 0.5

        # Normalize the distance-based reward between -1 and 1
        d_min = 0  # Minimum possible distance
        d_max = ((self.grid_size[0] - 1) ** 2 + (self.grid_size[1] - 1) ** 2) ** 0.5  # Maximum possible distance
        scaled_distance = 2 * (distance_to_goal - d_min) / (d_max - d_min) - 1

        return -1 #int(distance_to_goal)

    def nearest_valid_action(self, action):
        # Define the valid actions
        valid_actions = [0, 1, 2, 3, 4, 5, 6, 7]
        
        # Find the nearest valid action
        nearest_action = min(valid_actions, key=lambda x: abs(x - action))
        
        return nearest_action
    
    
    def step(self, action):
        done = False
        truncated = False
        row, col = self.agent_position
        #print(self.agent_position)
        new_row, new_col = row, col


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

        self.current_step +=1
        # Check if the new position is valid
        if not self._is_collision(new_position):
            self.agent_position = new_position

        # Calculate the reward based on the new position
        reward = self._calculate_reward(new_position)
        
        if reward == 30:  # Positive reward for reaching the goal
            done = True
            
        if self.current_step>=self.max_steps:
            done = True
            truncated= True
        #if truncated and self.agent_position != self.goal_position:
        #    reward = -1  # Negative reward for reaching the maximum steps without reaching the goal

        return self._get_observation(), reward, done,truncated, {}
    #return self._get_observation(), reward, done, truncated, {}
    
    

    def get_dataset(self):
        return self.dataset
    
    def _is_collision(self, position):
        # Check if Agent hits obstacle
        if position in self.obstacle_positions:
            return True

        # Check if the agent hits the wall
        if position[0] < 0 or position[0] >= self.num_rows or \
                position[1] < 0 or position[1] >= self.num_cols:
            return True

        return False

    def close(self):
        self.dataset.close()
    
    def is_position_valid(self, position, obstacle_positions):
        for obstacle_pos, obstacle_size in obstacle_positions:
            obs_row_start, obs_col_start = obstacle_pos
            obs_row_end = obs_row_start + obstacle_size[0]
            obs_col_end = obs_col_start + obstacle_size[1]
            
            row, col = position
            if obs_row_start <= row < obs_row_end and obs_col_start <= col < obs_col_end:
                return False
        return True
    
    
    def generate_valid_start_goal(self):
        possible_starts = [(1,1)]   #[(0,np.random.randint(1,3))]  # [(1,1)]
        possible_starts1 = [(1,6)]  #[(0,np.random.randint(5,7))] ## [(1,6)]
        #possible_ends = [(8,8), (8,1)]
        while True:
            #(4,0) #possible_starts[self.goalselector]
            obstacle_coordinates = [(6,1), (6,6)]
            topdowngen = 1#np.random.randint (0,2)
            startsel = np.random.randint (0,2)
            if topdowngen ==0:
                self.start_position = possible_starts1[0] #original possible_starts1[0]
                self.goal_position = (obstacle_coordinates[topdowngen][0]-1, obstacle_coordinates[topdowngen][1])
            #goal_position = possible_ends[topdowngen]
            else:    
                self.start_position = possible_starts[0] #original possible_starts[0]
                self.goal_position = (obstacle_coordinates[topdowngen][0]-1, obstacle_coordinates[topdowngen][1])
            #goal_position = (np.random.randint(self.num_rows - 2, self.num_rows), np.random.randint(self.num_rows - 2, self.num_rows))
            #if self.is_position_valid(self.start_position, obstacle_positions) and self.is_position_valid(self.goal_position, obstacle_positions):
            return self.start_position, self.goal_position
            
            
    def _generate_obstacle_positions(self):
        #generating obstacles

        obstacle_coordinates = [(6,1), (6,6)]
        num_obstacles = len(obstacle_coordinates)
        obstacle_positions = []

        for _ in range(num_obstacles):
            obstacle_row = 3
            obstacle_col = 3
            obstacle_size = (1, 1)

            obstacle_positions.append(((obstacle_coordinates[_]), obstacle_size))

        return obstacle_coordinates
    
    @property
    def _max_episode_steps(self):
        return self.max_episode_steps
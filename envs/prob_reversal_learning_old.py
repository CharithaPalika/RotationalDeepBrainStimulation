import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


class NonStationaryEnv():
    ''' Nonstationary environment where the reward probabilities or values change after certain trials.
    Rewards are sampled from ~N(mean, std) with specified probabilities.'''
    def __init__(self, num_arms, mean_reward, std, 
                 probabilities,
                 mean_rew_change=None, std_change=None, 
                 probabilities_change=None,
                 stationary=False, time_stamp_change=None,
                 change_type='both'):
        self.num_arms = num_arms
        self.mean_reward = mean_reward
        self.std = std
        self.probabilities = probabilities
        self.stationary = stationary
        
        # Parameters for change
        self.mean_rew_change = mean_reward if mean_rew_change is None else mean_rew_change
        self.std_change = std if std_change is None else std_change
        self.probabilities_change = probabilities if probabilities_change is None else probabilities_change
        self.time_stamp_change = time_stamp_change
        
        # What type of change to apply: 'reward', 'probability', or 'both'
        self.change_type = change_type

        # Check that arrays have correct shapes
        assert self.num_arms == self.mean_reward.shape[0] == self.std.shape[0], 'Invalid shape of mean_reward or std array'
        assert self.num_arms == self.probabilities.shape[0], 'Invalid shape of probabilities array'
        
        if not stationary and time_stamp_change is not None:
            if self.change_type in ['reward', 'both']:
                assert self.num_arms == self.mean_rew_change.shape[0] == self.std_change.shape[0], 'Invalid shape of mean_rew_change or std_change array'
            if self.change_type in ['probability', 'both']:
                assert self.num_arms == self.probabilities_change.shape[0], 'Invalid shape of probabilities_change array'
        
        # Generate reward probability timestamps
        self.rew_timestamps = self.create_rew_timepoints()
        
        # Shuffle each key in rew_timestamps
        for key in self.rew_timestamps:
            np.random.shuffle(self.rew_timestamps[key])
        
        # Pick counter for each arm
        self.counts = np.zeros((self.num_arms))
        
        # Create arms dictionary for rewards
        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))
        self.step_counter = 0

    def create_rew_timepoints(self, probs=None):
        """
        Create binary arrays representing when rewards are given based on probabilities
        """
        # Use current probabilities if none provided
        if probs is None:
            probs = self.probabilities
            
        # Number of columns
        num_columns = 10

        # Create binary array
        binary_array = np.zeros((len(probs), num_columns), dtype=int)

        for i, prob in enumerate(probs):
            # Calculate the number of 1s based on the probability
            num_ones = int(prob * num_columns)
            
            # Randomly choose `num_ones` indices to set as 1
            one_indices = np.random.choice(num_columns, num_ones, replace=False)
            binary_array[i, one_indices] = 1
        
        binary_dict = {i: row.tolist() for i, row in enumerate(binary_array)}
        
        return binary_dict

    def step(self, chosen_arm):
        if not self.stationary and self.step_counter >= self.time_stamp_change:
            # Apply changes based on change_type
            if self.change_type in ['reward', 'both']:
                self.arms = dict(enumerate(zip(self.mean_rew_change, self.std_change)))
            
            if self.change_type in ['probability', 'both']:
                self.rew_timestamps = self.create_rew_timepoints(self.probabilities_change)
                # Shuffle the new timestamps
                for key in self.rew_timestamps:
                    np.random.shuffle(self.rew_timestamps[key])
                # Reset counts when probabilities change
                self.counts = np.zeros((self.num_arms))
        
        # Get mean and std for the chosen arm
        arm_mean, arm_dev = self.arms[chosen_arm]
        
        # Check if reward should be given based on probability
        if self.rew_timestamps[chosen_arm][int(self.counts[chosen_arm])] == 1:
            # Reward associated with the arm
            rew = np.random.normal(arm_mean, arm_dev)
        else:
            rew = 0
            
        # Update counter for this arm
        self.counts[chosen_arm] += 1
        
        # If we've used all timestamps for this arm, reset and reshuffle
        if self.counts[chosen_arm] == 10:
            self.counts[chosen_arm] = 0
            np.random.shuffle(self.rew_timestamps[chosen_arm])
            
        self.step_counter += 1
        return rew
    
    def reset(self):
        # Reset arms to initial values
        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))
        
        # Reset reward timestamps based on initial probabilities
        self.rew_timestamps = self.create_rew_timepoints()
        for key in self.rew_timestamps:
            np.random.shuffle(self.rew_timestamps[key])
            
        # Reset counters
        self.counts = np.zeros((self.num_arms))
        self.step_counter = 0
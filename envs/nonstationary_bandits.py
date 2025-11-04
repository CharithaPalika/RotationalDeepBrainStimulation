import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class NonStationaryEnv():
    ''' Nonstationary environment where the mean reward changes after certain trails and rewards are sampled from ~N(mean, std)'''
    def __init__(self, num_arms, mean_reward, std, mean_rew_change, std_change, time_stamp_change = None):
        self.num_arms = num_arms
        self.mean_reward = mean_reward
        self.std = std
        self.mean_rew_change = mean_rew_change
        self.std_change = std_change
        self.time_stamp_change = time_stamp_change

        assert self.num_arms == self.mean_reward.shape[0] == self.std.shape[0] == self.mean_rew_change.shape[0] == self.std_change.shape[0], 'Invalid shape of mean_reward or std array'
        
        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))
        self.step_counter = 0

    def step(self, chosen_arm):
       
        if self.step_counter >= self.time_stamp_change:
            self.arms = dict(enumerate(zip(self.mean_rew_change, self.std_change)))
        
        arm_mean, arm_dev = self.arms[chosen_arm]
        self.step_counter += 1
        return np.random.normal(arm_mean, arm_dev)
    
    def reset(self):
    
        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))
        self.step_counter = 0     

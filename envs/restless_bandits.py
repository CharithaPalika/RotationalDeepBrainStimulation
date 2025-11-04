import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns



class RestlessBanditsEnv():
    ''' Restless bandit environment where the mean of the reward keeps drifting away irresoective of whether the arm is chosen or not'''
    def __init__(self, num_arms, mean_reward, std, 
                 lambda_decay = 0.9836/100, theta = 50/100, 
                 diffusive_noise_std = 2.8/100, diffusive_noise_mean = 0):
        self.num_arms = num_arms
        self.mean_reward_initial = mean_reward
        self.std_initial = std
        self.mean_reward = mean_reward
        self.std = std
        assert self.num_arms == self.mean_reward.shape[0] == self.std.shape[0], 'Invalid shape of mean_reward or std array'
        
        self.lambda_decay = lambda_decay
        self.theta = theta
        self.diffusive_noise_std = diffusive_noise_std
        self.diffusive_noise_mean = diffusive_noise_mean

        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))
        self.step_counter = 0

    def update_arm_mean(self):
        diffusive_noise = np.random.normal(self.diffusive_noise_mean, self.diffusive_noise_std,4)
        self.mean_reward = self.lambda_decay * self.mean_reward + (1 - self.lambda_decay) * self.theta + diffusive_noise
        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))

    def step(self, chosen_arm):
        arm_mean, arm_dev = self.arms[chosen_arm]
        self.step_counter += 1
        reward = np.random.normal(arm_mean, arm_dev)
        self.update_arm_mean()
        return reward
    
    def reset(self):
        assert self.num_arms == self.mean_reward_initial.shape[0] == self.std_initial.shape[0], 'Invalid shape of mean_reward or std array'
        self.arms = dict(enumerate(zip(self.mean_reward_initial, self.std_initial)))
        self.step_counter = 0    

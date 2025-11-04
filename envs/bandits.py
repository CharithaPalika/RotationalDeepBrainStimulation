import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


class BanditEnv():
    ''' Stationary environment where the mean reward is fixed and rewards are sampled from ~N(mean, std)'''
    def __init__(self,num_arms: int,mean_reward: np.ndarray, std: np.ndarray):
        self.num_arms = num_arms
        self.mean_reward = mean_reward
        self.std = std
        assert self.num_arms == self.mean_reward.shape[0] == self.std.shape[0], 'Invalid shape of mean_reward or std array'
        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))

    def step(self, chosen_arm):
        arm_mean, arm_dev = self.arms[chosen_arm]
        return np.random.normal(arm_mean, arm_dev)
    
    def plot_arms(self):
        arm_data = np.zeros((500,self.num_arms))
        for i, data in enumerate(zip(self.mean_reward, self.std)):
            mu,sigma = data
            arm_data[:,i] = np.random.normal(mu, sigma,size = (500))
        
        plt.title('Reward distribution for arms', fontsize = 9)
        plt.ylabel('Rewards distribution', fontsize = 8)
        plt.xlabel('Arms', fontsize = 8)
        plt.xticks(range(1,self.num_arms+1))
        plt.violinplot(arm_data, positions=range(1,self.num_arms+1))
        plt.show()
        plt.tight_layout()
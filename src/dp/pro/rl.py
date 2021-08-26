'''
Created on Aug 25, 2021

@author: immanueltrummer
'''
import argparse
import gym
from gym.spaces import Box, Discrete
import heapq
import numpy as np
import os
import pandas as pd
from stable_baselines3 import A2C

class CordulaGym(gym.Env):
    """ RL environment for detecting correlations. """
    
    step_s = 0.05
    s_per_pred = 26.0/11886
    
    def __init__(self, pred_path, data_scaling):
        """ Initializes environment with given input file.
        
        Args:
            pred_path: path to pre-generated predictions
            data_sacling: scaling factor for data processing
        """
        self.df = pd.read_csv(pred_path, sep=',')
        self.nr_pairs = self.df.shape[0]
        self.data_scaling = data_scaling
        self.pred_per_step = round(
            self.step_s * self.data_scaling/self.s_per_pred)
        self.next_pair = 0
        self.heap = []
        self.sim_time = 0.0
        self.sim_hits = 0
        self.log_hits = []
        self.observation_space = Box(low=0, high=1.0, shape=(2,))
        self.action_space = Discrete(3)
    
    def reset(self):
        return self._observe()
    
    def step(self, action):
        cur_nr_hits = 0
        if action == 0: # Apply NLP to next batch of column pairs
            nr_remaining = self.nr_pairs - self.next_pair
            batch_size = min(nr_remaining, self.pred_per_step)
            last_pair = self.next_pair + batch_size
            batch = self.df.iloc[self.next_pair:last_pair,:]
            for _, row in batch.iterrows():
                pred = row['predictions']
                label = row['labels']
                cost = row['time']
                heapq.heappush(self.heap, (-pred, label, cost))
            
            self.next_pair += batch_size
            self.sim_time += batch_size * self.s_per_pred
            
        elif action == 1: # Process items without predictions
            deadline = self.sim_time + self.step_s
            while self.sim_time < deadline and self.next_pair < self.nr_pairs:
                row = self.df.loc[self.next_pair]
                label = row['labels']
                cost = row['time']
                self.sim_time += cost
                self.next_pair += 1
                if label:
                    cur_nr_hits += 1
            
        elif action == 2: # Process items from heap
            deadline = self.sim_time + self.step_s
            while self.sim_time < deadline and self.heap:
                pred, label, cost = heapq.heappop(self.heap)
                self.sim_time += cost
                if label:
                    cur_nr_hits += 1
        
        self.sim_hits += cur_nr_hits
        self._log()
        return self._observe(), cur_nr_hits, False, {}
    
    def _log(self):
        self.log_hits.append((self.sim_time, self.sim_hits))
    
    def _observe(self):
        heap_size = len(self.heap) / self.nr_pairs
        if heap_size > 0:
            pred, _, _ = self.heap[0]
        else:
            pred = 0
        return np.array([heap_size, -pred])


if __name__ == '__main__':
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to input prediction file')
    args = parser.parse_args()
    
    for scale in [1, 10, 100, 1000]:
        env = CordulaGym(args.in_path, scale)
        model = A2C(
            'MlpPolicy', env, verbose=True, 
            gamma=1.0, normalize_advantage=True)
        model.learn(total_timesteps=20000)
        
        log_df = pd.DataFrame(env.log_hits, columns=['ctime','chits'])
        log_df['crows'] = 0
        log_df['step'] = 0
        log_df.to_csv(f'results/alltables_F{scale}_rl.csv')
    
    #
    #
    # with open('results/alltablesrl.csv', 'w') as file:
        # for 
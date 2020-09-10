# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:38:44 2019

@author: sahil.d
"""

import gym
import numpy as np

env = gym.make("Taxi-v2").env

print(env.action_space)
#> Discrete(6)
print(env.observation_space)
#> Discrete(500)


######## Q-Learning
alpha = 0.8
n_episodes = 1000
gamma = 0.9
epsilon = 0.9999

n_states = env.observation_space.n
n_actions = env.action_space.n

np.random.seed(seed = 2019)
#Q = 2*np.ones((n_states, n_actions)) 
Q = 0*np.random.random_sample((n_states, n_actions))
Q[0] = 0*Q[0]

#visited = epsilon*np.ones((n_states, n_actions))
eps = epsilon*np.ones(n_states)

for i_episode in range(n_episodes):
    from_state = env.reset()
    is_terminal = from_state==0
    
    while is_terminal == False:
        
        if np.random.rand() < eps[from_state]:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[from_state])
            
        to_state, reward, is_terminal, info = env.step(action)
        
        Q[from_state][action] = (1-alpha)*Q[from_state][action] + \
                    alpha*(reward + gamma*max([Q[to_state][a] for a in range(n_actions)]))
                    
        eps[from_state] = eps[from_state]/(eps[from_state] + 0.1)
        
        from_state = to_state
        
env.close()

print(Q[462][4], "vs -11.374402515")
print(Q[398][3], "vs 4.348907")
print(Q[253][0], "vs -0.5856821173")
print(Q[377][1], "vs 9.683")
print(Q[83][5], "vs -12.8232660372")


print(Q[236][4])
print(Q[188][2])
print(Q[306][0])
print(Q[216][0])
print(Q[38][0])
print(Q[329][5])
print(Q[276][1])
print(Q[21][0])
print(Q[71][4])
print(Q[316][0])
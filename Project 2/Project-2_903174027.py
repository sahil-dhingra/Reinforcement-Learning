# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 05:31:53 2019

@author: sahil.d
"""


import gym
import numpy as np
import pybox2d
import Box2D
import random
from copy import deepcopy
import time
import collections
from bayes_opt import BayesianOptimization as bo

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from keras.optimizers import SGD

env = gym.make("LunarLander-v2").env

print(env.action_space)
#> Discrete(6)
print(env.observation_space)
#> Discrete(500)


######## Q-Learning

#alpha = 0.1
#C = 16
#n_states = env.observation_space
#n_actions = env.action_space.n
def dqn(learning_rate, decay):
    
    n_episodes = 900
    gamma = 0.99
    epsilon = 1
    decay_rate = decay
    T = 1024
    batch_size = 64
    N = 50000
    eps_min = 0.02
    
    
    np.random.seed(seed = 2024)
    Q = Sequential()
    Q.add(Dense(8, activation='relu', input_shape=(8,)))
    Q.add(Dense(8, activation='relu'))
    Q.add(Dense(4, activation='linear'))
    
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    Adam = optimizers.Adam(lr = learning_rate)
    Q.compile(loss='mean_squared_error', optimizer=Adam)
    
    Q_ = deepcopy(Q)
    
    buffer = collections.deque([], N)
    results = []
    
    random.seed(2024)
    r1 = random.randint(0,1000)
    env.seed(r1)
    
    for i_episode in range(n_episodes): 
        start = time.process_time()
        from_state = env.reset()
        is_terminal = False
        epsilon = eps_min + (1-eps_min)*np.exp(-decay_rate*i_episode)
        #t = 0
        score = 0
        np.random.seed(seed = i_episode + 1)
        random.seed(i_episode + 1)
        for t in range(T):
            if np.random.rand() < epsilon:
                action = random.randint(0,3)
            else:
                action = np.argmax(Q.predict_on_batch(np.array([from_state])))
            
            to_state, reward, is_terminal, info = env.step(action)
            
            buffer.append(np.array([from_state, to_state, action, reward, is_terminal]))
            
            from_state = to_state
            
            score = score + reward
            
            if i_episode>0:
                mb = random.sample(list(buffer), batch_size)
                from_states = np.vstack(np.array(mb).T[0].T)
                to_states = np.vstack(np.array(mb).T[1].T)
                actions = np.array(mb).T[2].T
                rewards = np.array(mb).T[3].T
                m_terminal = np.argwhere(np.array(mb).T[4].T).flatten()
    
               #y = np.zeros((batch_size,4))
                y = Q.predict_on_batch(from_states)
                
                y[list(range(batch_size)), list(actions)] = rewards + \
                                    gamma*np.max(Q_.predict_on_batch(to_states), axis = 1)
           
                y[list(m_terminal), list(actions[m_terminal])] = rewards[m_terminal]
    
                Q.train_on_batch(from_states, y)
               
                if t%8 == 0:
                    Q_.set_weights(Q.get_weights())
            
                if is_terminal or t==T-1:
                    results.append(score)
                    break
        
        env.close()

        print(i_episode, "reward =", round(reward,1), "score", round(score,1), "T =", round(time.process_time() - start, 1), "i =", t, "eps=", round(epsilon,3))
    return results, Q


# Running the train function with different lr and decay values
lr = [0.0006]
decay = [0.01]
grid_items = []
grid_results = []
models = []
for i in lr:
    for j in decay:
        print(i, j)
        grid_items.append([i, j])
        out, model = dqn(i, j)
        grid_results.append(out)
        models.append(model)
        
        
#Testing
Q = models[0]
env.seed(1)
test = []
for i in range(100):
    from_state = env.reset()
    is_terminal = False
    score = 0
    while not is_terminal:
        action = np.argmax(Q.predict_on_batch(np.array([from_state])))
        #action = env.action_space.sample()
        to_state, reward, is_terminal, info = env.step(action)
        from_state = to_state
        score = score + reward
        #env.render()
    #test.append(score)
    env.close()


## Plotting
import matplotlib.pyplot as plt

#Train
plt.plot(results, 'b')
train_ma = [None]*100 + [np.mean(results[i:i+100]) for i in range(len(results)-100)]
plt.plot(train_ma, 'r', label = 'Best α')
plt.ylim(-400, 400)
plt.title('Total rewards over episodes for Training')
plt.xlabel('Episode')
plt.ylabel('Total Rewards in the episode')
plt.rcParams["font.size"] = "30"
plt.rcParams["figure.figsize"] = [24,12]
plt.savefig('train.jpg')

#Test
plt.plot(test, 'b')
plt.ylim(0, 400)
plt.title('Total rewards over episodes for Test Trials')
plt.xlabel('Episode')
plt.ylabel('Total Rewards in the episode')
plt.rcParams["font.size"] = "30"
plt.rcParams["figure.figsize"] = [24,12]
plt.savefig('test.jpg')
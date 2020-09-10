# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:45:41 2019

@author: sahil.d
"""
N = 21, isBadSide = {1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,1,0}, Output: 7.3799
N = 22, isBadSide = {1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0}, Output: 6.314
N = 6, isBadSide = {1,1,1,0,0,0}, Output: 2.5833

import numpy as np
from itertools import product
from copy import deepcopy

N=6
isBadSide = [1,1,1,0,0,0]

N=22
isBadSide = [1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0]

N=21
isBadSide = [1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,1,0]

N=18
isBadSide = [0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,0,0,0]

N=22
isBadSide=[0,1,0,1,0,0,1,1,0,0,1,1,0,1,1,1,1,0,1,1,0,1]

N=21
isBadSide=[0,1,1,1,0,0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,0]

N=23
isBadSide=[0,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,0,1]

N=3
isBadSide = [0,0,1]

N=14
isBadSide = [0,0,0,1,1,0,1,0,0,1,1,0,0,1]

N=14
isBadSide = [0,0,1,1,1,0,1,1,1,1,0,1,0,0]

N=25
isBadSide = [0,1,0,1,1,1,0,1,1,0,0,0,1,1,1,1,0,1,1,0,1,0,1,0,0]

N=25
isBadSide = [0,1,1,0,0,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,0,0,1]

N=8
isBadSide = [0,1,1,0,0,0,1,1]

isGoodSide = list(1-np.array(isBadSide))
prob = sum(isBadSide)/N
die = list(range(N+1))[1:]

reward = [a*b for a,b in zip(die, isGoodSide) if b>0]
reward_sum = sum(reward)
reward_states = [0]
gamma = 1

#reward_all = list(set([sum(s) for s in product(reward[::1], repeat=2) if sum(s)<=2*N]))

def available_actions(state):
    if state[1] == 1 or state[0]*prob >= reward_sum/N:
        action = [0]
    else:
        action = [0, 1]
    return action

def next_states(state, action):
    states = []
    if action == 0:
        states = [state[0], 1]
    if action == 1:
        for i in range(len(reward)):
            states.append([state[0] + reward[i], 0])
    return states

def generate_states(reward_states):
    for j in set(reward_states):        
        if j*prob<reward_sum/N:
            for i in range(len(reward)):
                reward_states.append(j + reward[i])
    states_set = list(set(reward_states))
    return states_set
      
def states_list(states_set):  
    states = []
    for i in range(len(states_set)):
        states.append([states_set[i], 0])
        states.append([states_set[i], 1])
    return states

states = states_list(generate_states(generate_states(reward_states)))

def policy(state):
    if state[0]*prob<reward_sum/N:
        action = 1
    else:
        action = 0
    return action

def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        for j in range(len(pattern)):
            if mylist[i] == pattern[j]:
                matches.append(i)
    return matches


space = len(states)

value = np.zeros(space)
value_new = np.ones(space)
Q = np.zeros((space, 2))

while max(abs(value_new - value))>0.001:
    value = deepcopy(value_new)
    for s in range(space):
        for a in available_actions(states[s]):
            if a==1:
                Q[s,1] = (reward_sum/N - prob*states[s][0]) + gamma*sum([value[i]/N for i in subfinder(states, next_states(states[s], 1))])
            else:
                Q[s,1] = 0
            Q[s,0] = 0
        value_new[s] = Q[s, policy(states[s])]

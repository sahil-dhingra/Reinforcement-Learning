import numpy as np
import random
from Soccer_game import soccer
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from itertools import chain

sc = soccer()

def Q_learning(gamma, alpha, alpha_min, decay_rate, T, seed):
    # Initialize values
    V = np.ones((2,8,8,2))
    Q = np.ones((2,8,8,2,5))
    
    random.seed(seed)
    
    error = []
    epsilon = 1
    gamma = gamma
    alpha = alpha
    alpha_min = alpha_min
    decay_rate = decay_rate
    T = T
    moves = [[0,-1],
             [-1,0],
             [0, 0],
             [1, 0],
             [0, 1]]
    
    for t in range(T):
        
        #Decay alpha
        alpha = alpha_min + (alpha - alpha_min)*np.exp(-decay_rate*t)
        epsilon = alpha_min + (epsilon - alpha_min)*np.exp(-20*decay_rate*t)
        
        #Reset game
        sc.reset()
        
        action_A, action_B = random.randrange(5), random.randrange(5)
        
        while sc.is_over != 1:
            # Generate random states
            
            actions = [action_A, action_B]
            
            # Get current state
            a, b, ball = sc.i_state()
            
            # Execute actions for A & B
            sc.play(action_A, action_B)
            
            # Get rewards
            R = sc.get_reward()
            
            # Get new states
            a_, b_, ball_ = sc.i_state()
            
            # Old Q
            q_old = Q[0, 5, 3, 1, 0]
            
            # Update V & Q using CE-Q
            # Player A
            V[0, a_, b_, ball_] = np.max(Q[0, a_, b_, ball_])
            Q[0, a, b, ball, actions[0]] = \
                                    (1-alpha)*Q[0, a, b, ball, actions[0]] + \
                                    alpha*((1-gamma)*R[0] + gamma*V[0, a_, b_, ball_])
                    
            # Player B
            V[1, b_, a_, ball_] = np.max(Q[1, b_, a_, ball_])
            Q[1, b, a, ball, actions[1]] = \
                                    (1-alpha)*Q[1, b, a, ball, actions[1]] + \
                                    alpha*((1-gamma)*R[1] + gamma*V[1, b_, a_, ball_])
            
            # Epsilon-action selection
            if np.random.rand() < epsilon:
                action_A, action_B = random.randrange(5), random.randrange(5)
            else:
                action_A, action_B = Q[0, a_, b_, ball_].argmax(), Q[1, b_, a_, ball_].argmax()

            q_new = Q[0, 5, 3, 1, 0] 
            error.append(abs(q_new - q_old))
                    
    return V, Q, error


# Running the algorithm
a, dr, seed = 0.2, 0.00000000015, 102
V, Q, error = Q_learning(gamma=0.9, alpha=a, alpha_min=0.001, decay_rate=dr, T=200000, seed=seed)

## Plotting
error_z = [i for i in range(len(error)) if error[i]!=0]
err = np.interp(list(range(len(error))), error_z, [error[i] for i in error_z])

#Train
x = 40
plt.rcParams["font.size"] = '{}'.format(1.5*x)
plt.rcParams["figure.figsize"] = [1.25*x,x]
plt.plot(err[:1000000], 'k')
plt.ylim(0, 0.5)
plt.title('Q-Learning')
plt.xlabel('Simulation Iteration')
plt.ylabel('Q-value difference')
plt.savefig('Q-Learning on-policy {}, {}, {} (gamma decay factor 20).jpg'.format(a, dr, seed))
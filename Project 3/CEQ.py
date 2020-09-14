import numpy as np
import random
from Soccer_game import soccer
from scipy.optimize import linprog
from cvxopt import matrix, solvers
from itertools import chain

# GLPK tolerances and other settings
solvers.options['abstol'] = 1e-4 # Default is 1e-7
solvers.options['reltol'] = 1e-4 # Default is 1e-6
solvers.options['feastol'] = 1e-4 # Default is 1e-7
solvers.options['show_progress'] = False
solvers.options['glpk'] = {'msg_lev' : 'GLP_MSG_OFF', 'LPX_K_MSGLEV': 0}

sc = soccer()

def ceq(gamma, alpha, alpha_min, decay_rate, T, seed):
    sc = soccer()
    
    random.seed(seed)
    # Initialize values
    V = np.ones((2,8,8,2))
    Q = np.ones((2,8,8,2,5,5))
    error = []

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

        alpha = alpha_min + (alpha - alpha_min)*np.exp(-decay_rate*t)
        sc.reset()
        action_A, action_B = random.randrange(5), random.randrange(5)
        while sc.is_over != 1:
            # Generate random states
            action_A, action_B = random.randrange(5), random.randrange(5)
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
            q_old = Q[0, 5, 3, 1, 0, 2]

            # Update V & Q using CE-Q
            V[0, a_, b_, ball_], V[1, b_, a_, ball_] = lp_ceq(Q[0, a_, b_, ball_], Q[1, b_, a_, ball_])
            Q[0, a, b, ball, actions[0], actions[1]] = \
                                    (1-alpha)*Q[0, a, b, ball, actions[0], actions[1]] + \
                                    alpha*((1-gamma)*R[0] + gamma*V[0, a_, b_, ball_])

            # Player B
            Q[1, b, a, ball, actions[1], actions[0]] = \
                                    (1-alpha)*Q[1, b, a, ball, actions[1], actions[0]] + \
                                    alpha*((1-gamma)*R[1] + gamma*V[1, b_, a_, ball_])

            q_new = Q[0, 5, 3, 1, 0, 2] 
            error.append(abs(q_new - q_old))  
            
        if t%100==0:
            print(t)
    return V, Q, error


def lp_ceq(Q1_state, Q2_state):
    G=[]
    for i in range(5):
        for j in chain(range(i+1,5), range(0,i)):
            lp_A = np.zeros((5,5))
            lp_A[i,:] =  - (Q1_state[i,:] - Q1_state[j,:])
            G.append(list(lp_A.flatten()))
            lp_B = np.zeros((5,5))
            lp_B[:,i] =  - (Q2_state[i,:] - Q2_state[j,:])
            G.append(list(lp_B.flatten()))

    for i in range(25):
        G.append(i*[0.] + [-1.] + (24-i)*[0.])
    G = matrix(np.array(G))
    h = matrix(65*[0.])
    A = matrix(matrix(25*[[1.]]))
    b = matrix(1.)
    c = matrix(25*[1.])
    sol = solvers.lp(c, G, h, A, b, solver = 'glpk')
    if sol['x'] is None:
        sol = solvers.lp(c, G, h, A, b)
    policy = np.array(sol['x']).T[0]
    policy = policy.reshape((5,5)) 
    v1 = np.sum(policy*Q1_state)
    v2 = np.sum(policy*Q2_state)
    return v1, v2


# Running the algorithm
a, dr, seed = 0.2, 0.0000000005, 102
V, Q, error = ceq(gamma=0.9, alpha=a, alpha_min=0.001, decay_rate=dr, T=200000, seed=seed)

## Plotting
error_z = [i for i in range(len(error)) if error[i]!=0]
err = np.interp(list(range(len(error))), error_z, [error[i] for i in error_z])

#Train
x = 40
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = '{}'.format(1.5*x)
plt.rcParams["figure.figsize"] = [1.25*x,x]
plt.plot(err, 'k')
plt.ylim(0, 0.5)
plt.title('CE-Q')
plt.xlabel('Simulation Iteration')
plt.ylabel('Q-value difference')
plt.savefig('CE-Q {}, {}, {}.jpg'.format(a, dr, seed))

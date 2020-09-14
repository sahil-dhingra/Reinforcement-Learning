import random
from copy import deepcopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


lambdas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

start = [0, 0, 0, 1, 0, 0, 0]

seed = 10

#Function to generate a single sequence
def generate_sequence(seed):
    random.seed(seed)
    old_location = start.index(1)
    seq = [start]
    state = deepcopy(start)
    new_location = 0
    while old_location!=0 and old_location!=6:
        new_location = [old_location-1, old_location+1][random.randint(0,1)]
        state[old_location] = 0
        state[new_location] = 1
        seq.append(deepcopy(state))
        old_location = state.index(1)
    return seq

# Create 100 training sets with 10 sets of sequences each
training_set = [0]*100
p_seed = 3
random.seed(seed)
for i in range(100):
    seeds = list(range(1,101))
    random.seed(seed)
    random.shuffle(seeds)
    training_set[i] = [generate_sequence(p_seed*100*seeds[i])]
    j = 1
    while j<10:
        training_set[i].append(generate_sequence(p_seed*100*seeds[i]+j))
        j = j+1

#Function to compute RMSE
def rmse(prediction):
    pred = prediction[1:6]
    actual = [1/6, 2/6, 3/6, 4/6, 5/6]
    rmse = np.sqrt(np.mean(np.square(np.subtract(pred, actual))))
    return rmse


####################
# Experiment 1
####################
    
alpha = 0.01
# Experiment 1 - update weights after the full sequence
def experiment_1(train_set, alpha, lambda_):
    w = [0, 0, 0, 0, 0, 0, 0]
    delta_w = [1, 1, 1, 1, 1, 1, 1]
    while abs(max(delta_w))>0.0001:
        delta_w = [0, 0, 0, 0, 0, 0, 0]
        for j in range(10):
            sequence = train_set[j]
            n = len(sequence)
            z = [0, 1][sequence[n-1][6]==1]  
            e_t = sequence[0]
            for i in range(n-2):
                delta_w = delta_w + np.multiply(alpha*(np.dot(w, sequence[i+1]) - 
                                                       np.dot(w, sequence[i])), e_t)
                e_t = sequence[i+1] + np.multiply(lambda_, e_t)
            delta_w = delta_w + np.multiply(alpha*(z - np.dot(w, sequence[n-2])), e_t)
        w = w + delta_w
    return w

error = [[], [], [], [], [], [], []]
for j in range(len(lambdas)):
    for i in range(100):
        predict = experiment_1(training_set[i], 0.01, lambdas[j])
        error[j].append(rmse(predict))

# Output   
fig3 = list(map(lambda x: np.mean(x), error))





#########################
## Experiment 2
#########################

lambdas_2 = list(map(lambda x: x/10, list(range(0, 11))))
alphas = list(map(lambda x: x/20, list(range(0, 13))))
alphas[0] = 0.01

#Define Experiment 2 - w updated after each sequence without convergence
def experiment_2(train_set, alpha, lambda_):
    w = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    for j in range(10):
        delta_w = [0, 0, 0, 0, 0, 0, 0]
        sequence = train_set[j]
        n = len(sequence)
        z = [0, 1][sequence[n-1][6]==1]        
        e_t = sequence[0]
        for i in range(n-2):
            delta_w = delta_w + np.multiply(alpha*(np.dot(w, sequence[i+1]) - 
                                                   np.dot(w, sequence[i])), e_t)
            e_t = sequence[i+1] + np.multiply(lambda_, e_t)
        delta_w = delta_w + np.multiply(alpha*(z - np.dot(w, sequence[n-2])), e_t)
        w = w + delta_w
    return w

error2 = [[]*11]
for i in range(len(error2)):
    for j in range(len(alphas)):
        error2[i].append([])
    
for j in range(len(lambdas_2)):
    for k in range(len(alphas)):
        for i in range(100):
            predict2 = experiment_2(training_set[i], alphas[k], lambdas_2[j])
            error2[j][k].append(rmse(predict2))

rmse2 = [[]*11]
for j in range(len(lambdas_2)):
    for k in range(len(alphas)):
        rmse2[j].append(np.mean(error2[j][k]))


#######################
#        Plots
#######################
        
#plot figure 3
plt.scatter(lambdas,fig3)
plt.ylim(0.125, 0.185)
plt.title('Figure 3')
plt.xlabel('λ')
plt.ylabel('RMSE')
plt.rcParams["figure.figsize"] = [6.4, 6.4]
plt.plot(lambdas,fig3, 'k', label = 'α = 0.01')
plt.savefig('fig3.jpg')

#Plot figure 4
fig4 = list(np.array(rmse2)[[0,3,8,10]])
plt.scatter(alphas, fig4[0])
plt.scatter(alphas, fig4[1])
plt.scatter(alphas, fig4[2])
plt.scatter(alphas, fig4[3])
plt.ylim(0, 0.7)
plt.xlim(0, 0.46)
plt.title('Figure 4')
plt.xlabel('λ')
plt.ylabel('RMSE')
plt.rcParams["figure.figsize"] = [6.4, 6.4]
l1, = plt.plot(alphas,fig4[0], 'k', label = 'λ = 0  ')
l2, = plt.plot(alphas,fig4[1], 'r', label = 'λ = 0.3')
l3, = plt.plot(alphas,fig4[2], 'g', label = 'λ = 0.8')
l4, = plt.plot(alphas,fig4[3], 'b', label = 'λ = 1.0')
plt.legend([l1, l2, l3, l4], ['λ = 0  ', 'λ = 0.3', 'λ = 0.8', 'λ = 1'])
plt.savefig('fig4.jpg')


fig5 = [min(rmse2[j]) for j in range(11)]

#Plot figure 5
plt.scatter(lambdas_2, fig5)
plt.ylim(0.09, 0.19)
plt.title('Figure 5')
plt.xlabel('λ')
plt.ylabel('RMSE')
plt.rcParams["figure.figsize"] = [6.4, 6.4]
plt.plot(lambdas_2, fig5, 'k', label = 'Best α')
plt.savefig('fig5.jpg')

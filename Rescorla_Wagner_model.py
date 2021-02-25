import random 
import numpy as np 
import matplotlib.pyplot as plt

#Problem 1 : The Rescorla-Wagner model
#Classical conditioning

def stimulus_array(nb_of_trials) :

    stimulus = np.ones(nb_of_trials)

    return(stimulus)

def rewards_array(nb_of_trials) : 

    rewards = np.concatenate((np.ones(nb_of_trials // 2), np.zeros(nb_of_trials // 2)), axis = None)

    return(rewards)

def Rescorla_Wagner_model(nb_of_trials, epsilon) : 
    w = 0
    delta = 0
    prediction = np.zeros(nb_of_trials)
    reward = rewards_array(nb_of_trials)
    stimulus = stimulus_array(nb_of_trials)
    
    for i in range(nb_of_trials) :
        delta = reward[i] - prediction[i-1]
        w = w + epsilon*delta*stimulus[i]
        prediction[i] = w*stimulus[i]

    return(prediction)

def Rescorla_Wagner_model_random(nb_of_trials, epsilon, prop_success) : 
    random.seed(25)
    w = 0
    delta = 0
    prediction = np.zeros(nb_of_trials)
    reward = np.random.binomial(size=nb_of_trials, n=1, p= prop_success)
    stimulus = stimulus_array(nb_of_trials)
    
    for i in range(nb_of_trials) :
        delta = reward[i] - prediction[i-1]
        w = w + epsilon*delta*stimulus[i]
        prediction[i] = w*stimulus[i]

    return(prediction)

def Rescorla_Wagner_model_2_stimuli_w1(nb_of_trials, epsilon) : 
    w_1 = np.zeros(nb_of_trials)
    w_2 = np.zeros(nb_of_trials)
    delta = 0
    prediction = np.zeros(nb_of_trials)
    reward = np.ones(nb_of_trials)
    stimulus_1 = stimulus_array(nb_of_trials)
    stimulus_2 = np.concatenate((np.zeros(nb_of_trials // 2), np.ones(nb_of_trials // 2)), axis = None)
    
    for i in range(nb_of_trials) :
        delta = reward[i] - prediction[i-1]
        w_1[i] = w_1[i-1] + epsilon*delta*stimulus_1[i]
        w_2[i] = w_2[i-1] + epsilon*delta*stimulus_2[i]
        prediction[i] = w_1[i]*stimulus_1[i] + w_2[i]*stimulus_2[i]

    return(w_1)

def Rescorla_Wagner_model_2_stimuli_W2(nb_of_trials, epsilon) : 
    w_1 = np.zeros(nb_of_trials)
    w_2 = np.zeros(nb_of_trials)
    delta = 0
    prediction = np.zeros(nb_of_trials)
    reward = np.ones(nb_of_trials)
    stimulus_1 = stimulus_array(nb_of_trials)
    stimulus_2 = np.concatenate((np.zeros(nb_of_trials // 2), np.ones(nb_of_trials // 2)), axis = None)
    
    for i in range(nb_of_trials) :
        delta = reward[i] - prediction[i-1]
        w_1[i] = w_1[i-1] + epsilon*delta*stimulus_1[i]
        w_2[i] = w_2[i-1] + epsilon*delta*stimulus_2[i]
        prediction[i] = w_1[i]*stimulus_1[i] + w_2[i]*stimulus_2[i]

    return(w_2)

def Rescorla_Wagner_model_2_stimuli_overshW1(nb_of_trials, epsilon) : 
    w_1 = np.zeros(nb_of_trials)
    w_2 = np.zeros(nb_of_trials)
    delta = 0
    prediction = np.zeros(nb_of_trials)
    reward = np.ones(nb_of_trials)
    stimulus_1 = stimulus_array(nb_of_trials)
    stimulus_2 = stimulus_array(nb_of_trials)

    for i in range(nb_of_trials) :
        delta = reward[i] - prediction[i-1]
        w_1[i] = w_1[i-1] + epsilon*delta*stimulus_1[i]
        w_2[i] = w_2[i-1] + epsilon*delta*stimulus_2[i]
        prediction[i] = w_1[i]*stimulus_1[i] + w_2[i]*stimulus_2[i]

    return(w_1)

def Rescorla_Wagner_model_2_stimuli_overshW2(nb_of_trials, epsilon) : 
    w_1 = np.zeros(nb_of_trials)
    w_2 = np.zeros(nb_of_trials)
    delta = 0
    prediction = np.zeros(nb_of_trials)
    reward = np.ones(nb_of_trials)
    stimulus_1 = stimulus_array(nb_of_trials)
    stimulus_2 = stimulus_array(nb_of_trials)
    
    for i in range(nb_of_trials) :
        delta = reward[i] - prediction[i-1]
        w_1[i] = w_1[i-1] + epsilon*delta*stimulus_1[i]
        w_2[i] = w_2[i-1] + epsilon*delta*stimulus_2[i]
        prediction[i] = w_1[i]*stimulus_1[i] + w_2[i]*stimulus_2[i]

    return(w_2)


Xmax = 60
Npoints = 60
x = np.linspace(0, Xmax, Npoints)

#figure 1
plt.figure()
y_1 = rewards_array(60)
plt.plot(x, y_1, 'o', label='Reward')
plt.xlabel("trial")
plt.ylabel("Probability of presence of reward")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_1.pdf')
plt.show()

#figure 2
plt.figure()
y_2 = stimulus_array(60)
plt.plot(x, y_2, 'o', label='Stimuli')
plt.xlabel("trial")
plt.ylabel("Probability of presence of stimulus")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_2.pdf')
plt.show()

#figure 3
plt.figure()
y1 = Rescorla_Wagner_model(60, 0.1)
plt.plot(x, y1, 'o', label='epsilon = 0.1')
plt.xlabel("trial")
plt.ylabel("prediction of the probability of reward")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_3.pdf')
plt.show()

#figure 4
plt.figure()
y4 = Rescorla_Wagner_model(60, 0.1)
y5 = Rescorla_Wagner_model(60, 0.3)
y6 = Rescorla_Wagner_model(60, 0.5)
y7 = Rescorla_Wagner_model(60, 0.9)
plt.plot(x, y4, 'o', label="epsilon = 0.1")
plt.plot(x, y5, 'o', label="epsilon = 0.3")
plt.plot(x, y6, 'o', label="epsilon = 0.5")
plt.plot(x, y7, 'o', label="epsilon = 0.9")
plt.xlabel("trial")
plt.ylabel("prediction of the probability of reward")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_4.pdf')
plt.show()

#Partial conditioning

#Figure 5
trial = np.zeros(60)

for i in range(60):
    trial[i] = np.random.binomial(1, 0.4, 1)

plt.figure()
plt.plot(x, trial, 'o')
plt.xlabel("trial")
plt.ylabel("presence of reward")
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_4.pdf')
plt.show()


#Figure 6
plt.figure()
y9 = Rescorla_Wagner_model_random(60, 0.1, 0.4)
plt.plot(x, y9, 'o')
plt.xlabel("trial")
plt.ylabel("prediction of the propability of reward")
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_5.pdf')
plt.show()

#Blocking

#Figure 7
plt.figure()
y10 = Rescorla_Wagner_model_2_stimuli_w1(60, 0.1)
y11 = Rescorla_Wagner_model_2_stimuli_W2(60, 0.1)
plt.plot(x, y10, 'o', label = 'W_1')
plt.plot(x, y11, 'o', label = 'W_2')
plt.xlabel("trial")
plt.ylabel("prediction of the propability of reward")
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_6.pdf')
plt.show()

#overshadowing

#Figure 8
plt.figure()
y12 = Rescorla_Wagner_model_2_stimuli_overshW1(60, 0.1)
y13 = Rescorla_Wagner_model_2_stimuli_overshW2(60, 0.2)
plt.plot(x, y12, 'o', label = "epsilon = 0.1")
plt.plot(x, y13, 'o', label="epsilon =0.2")
plt.xlabel("trial")
plt.ylabel("prediction of the propability of reward")
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_7.pdf')
plt.show()




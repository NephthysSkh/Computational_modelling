import random 
import collections
import numpy as np 
import matplotlib.pyplot as plt

random.seed(25)

def drift_diffusion_plot(m_a, m_b, x_0, delta_t, trials) :
    x = np.zeros(trials)
    x[0] = x_0
    sigma = 1/2

    for i in range(trials):
        x[i]= x[i-1] + (m_a - m_b)*delta_t + sigma*np.random.normal(loc=0, scale=1, size=None)*np.sqrt(delta_t)
        
    plt.figure()
    a = np.linspace(0, trials, trials)
    plt.plot(a, x)
    plt.xlabel("time [10th of miliseconds]")
    plt.ylabel("x")
    plt.tick_params("axis='both', labelsize=12")
    plt.savefig('figure_1.pdf')
    plt.show()

    return(x)


drift_diffusion_plot(1, 0.95, 0, 0.00001, 10000)


def drift_diffusion(m_a, m_b, x_0, delta_t, trials) :
    x = np.zeros(trials)
    x[0] = x_0
    sigma = 1/2

    for i in range(trials-1):
        x[i+1]= x[i] + (m_a - m_b)*delta_t + sigma*np.random.normal(loc=0, scale=1, size=None)*np.sqrt(delta_t)

    return(x)

n = 1000

outcome = np.zeros(n)
times_threshold = np.zeros(n)

for j in range(n):   
    x = drift_diffusion(1, 0.95, 0, 0.00001, 10000)
       
    for i in range(len(x)) :
        if x[i] < -0.1 :
            outcome[j] = 0
            times_threshold[j] = i
            break
        elif x[i] > 0.1 :
            outcome[j] = 1
            times_threshold[j] = i
            break

        outcome[j] = 2

print(collections.Counter(outcome))

outcome1 = []
outcome2 = []

# for i in range(len(outcome)) :
#     if outcome[i] :
#         outcome1.append(100 + int(times_threshold[i]))
#     else :
#         outcome2.append(100 + int(times_threshold[i]))

# print(outcome1)
# print(outcome2)

# axis_x = np.linspace(100, 10100, 10000)

# axis_y1 = np.zeros(10000)
# axis_y2 = np.zeros(10000)

# for i in range(10000) :
#     axis_y1[i] = outcome1.count(int(axis_x[i]))/len(outcome1)
#     axis_y2[i] = outcome2.count(int(axis_x[i]))/len(outcome2)

for i in range(len(outcome)) :
    if outcome[i] == 1 :
        outcome1.append(1 + int(times_threshold[i])//100)
    elif outcome[i] == 0:
        outcome2.append(1 + int(times_threshold[i])//100)

axis_x = np.linspace(1, 101, 100)

axis_y1 = np.zeros(100)
axis_y2 = np.zeros(100)

for i in range(100) :
    axis_y1[i] = outcome1.count(int(axis_x[i]))/len(outcome1)
    axis_y2[i] = outcome2.count(int(axis_x[i]))/len(outcome2)


plt.figure()
plt.plot(axis_x, axis_y1)
plt.xlabel("time [miliseconds]")
plt.ylabel("frequency of reaction time")
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_2.pdf')
plt.show()

plt.figure()
plt.plot(axis_x, axis_y2)
plt.xlabel("time [miliseconds]")
plt.ylabel("frequency of reaction time")
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_3.pdf')
plt.show()


#Question 3
"""def proba_outcome1(diff):
    p_A = []
    A = []
    B= []
    x = np.zeros(trials)
    x[0] = x_0
    sigma = 1/2

    for d in range(len(diff)):
        for i in range(9999):
            x[i+1]= x[i] + (diff)*delta_t + sigma*np.random.normal(loc=0, scale=1, size=None)*np.sqrt(delta_t)

        if x[i]==1 :
            A = A.append(1)
        elif x[i]==0:
            B = B.append(1)
        p_A = p_A.append(A/(A+B))

        print(p_A)
    return(p_A)


def proba_outcome1_theoric(diff):
    p_th = []
    beta = 2*0.1/0.5**2

    for d in diff :
        p_th = p_th.append(1/(1+np.exp(beta*d)))

    print(p_th)
    return(p_th)

diff = [-0.2, -0.1, 0, 0.1, 0.2]
proba_outcome1_theoric(diff)
proba_outcome1(diff)"""




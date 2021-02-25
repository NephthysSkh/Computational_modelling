import random 
import numpy as np 
import matplotlib.pyplot as plt

#Problem 2 : Simple decision strategy for flower sampling by bees

#plot p_b as a function of m_y - m_b with varying beta values
def plot_pb_as_funct_of_diff(beta) : 
    a = np.zeros(200)
    p_b = np.zeros(200)

    plt.figure()
    for b in beta : 
        for i in np.arange(0, 20, 0.1) :
            diff = int(i * 10)
            a[diff] = i - 10
            p_b[diff] = 1/(1 + np.exp(b*a[diff]))

        plt.plot(a, p_b, label = "beta = " + str(b))

    plt.xlabel("m_y - m_b")
    plt.legend()
    plt.ylabel("probability of bee landing on blue flower")
    plt.tick_params("axis='both', labelsize=12")
    plt.savefig('figure_1_bees.pdf')
    plt.show()

beta = [0, 0.25, 0.5, 0.75, 1]

plot_pb_as_funct_of_diff(beta)

#plot p_b as a function of beta with varying values for difference m_y - m_b

def plot_pb_as_funct_of_beta(diff) : 
    a = np.zeros(200)
    p_b = np.zeros(200)


    plt.figure()
    for d in diff : 
        for i in np.arange(0, 20, 0.1) :
            beta = int(i * 10)
            a[beta] = i - 10
            p_b[beta] = 1/(1 + np.exp(a[beta]*d))

        plt.plot(a, p_b, label = "m_y - m_b = " + str(d))

    plt.xlabel("beta")
    plt.legend()
    plt.ylabel("probability of bee landing on blue flower")
    plt.tick_params("axis='both', labelsize=12")
    plt.savefig('figure_2_bees.pdf')
    plt.show()



diff = [-10, -5, -1, 0, 1, 5, 10]
plot_pb_as_funct_of_beta(diff)

#types of bees
random.seed(25)

#Dumb bee

def plot_dumb_bee(beta, m_y_initial, m_b_initial, a) :
    trials = np.zeros(200)
    x = np.linspace(0, 200, 200)
    p_b = 1/(1 + np.exp(beta*(m_y_initial - m_b_initial)))

    for i in range(200) :
        trials[i] = np.random.binomial(1, p_b, 1)

    plt.figure()
    plt.plot(x, trials, 'o', label='1 = blue, 0 = yellow')
    plt.xlabel("trial")
    plt.ylabel("bee's choice blue")
    plt.tick_params("axis='both', labelsize=12")
    plt.savefig('figure_'+str(a)+'_bees.pdf')
    plt.show()

#beta = 0, explorative behaviour

plot_dumb_bee(0, 0, 5, 3)

#beta = 1, strongly exploitative behaviour

plot_dumb_bee(0.75, 0, 5, 4)



#smart bee
x = np.linspace(0, 200, 200)

def smart_bee_b(beta):    
    r_b = np.concatenate((np.ones(100)*8, np.ones(100)*2), axis=None)
    r_y = np.concatenate((np.ones(100)*2, np.ones(100)*8), axis=None)
    trials = np.zeros(200)
    m_b = np.zeros(200)
    m_y = np.zeros(200)
    p_b = np.zeros(200)
    p_y = np.zeros(200)
    epsilon = 0.2

    m_b[0] = 0
    m_y[0] = 5

    for i in range(199) : 
        p_b[i] = 1/(1 + np.exp(beta*(m_y[i] - m_b[i])))
        p_y[i] = 1 - p_b[i]
        m_b[i+1]= m_b[i] + epsilon*(r_b[i] - m_b[i])
        m_y[i+1] = m_y[i] + epsilon*(r_y[i] - m_y[i])

    return(p_b)


def smart_bee_y(beta):    
    r_b = np.concatenate((np.ones(100)*8, np.ones(100)*2), axis=None)
    r_y = np.concatenate((np.ones(100)*2, np.ones(100)*8), axis=None)
    trials = np.zeros(200)
    m_b = np.zeros(200)
    m_y = np.zeros(200)
    p_b = np.zeros(200)
    p_y = np.zeros(200)
    epsilon = 0.2

    m_b[0] = 0
    m_y[0] = 5

    for i in range(199) : 
        p_b[i] = 1/(1 + np.exp(beta*(m_y[i] - m_b[i])))
        p_y[i] = 1 - p_b[i]
        m_b[i+1]= m_b[i] + epsilon*(r_b[i] - m_b[i])
        m_y[i+1] = m_y[i] + epsilon*(r_y[i] - m_y[i])

    return(p_y)

p_b_1 = smart_bee_b(0.5)
p_y_1 = smart_bee_y(0.5)

plt.figure()
plt.plot(x, p_b_1, 'o', label = "blue flower")
plt.plot(x, p_y_1, 'o', label = "yellow flower")
plt.xlabel("trial")
plt.ylabel("probability of the bee landing on a flower")
plt.legend()
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_5_bees.pdf')
plt.show()

#different beta for the smart bee
r_b = np.concatenate((np.ones(100)*8, np.ones(100)*2), axis=None)
r_y = np.concatenate((np.ones(100)*2, np.ones(100)*8), axis=None)
trials = np.zeros(200)
m_b = np.zeros(200)
m_y = np.zeros(200)
epsilon = 0.2
m_b[0] = 0
m_y[0] = 5

p_b_0 = np.zeros(200)

for i in range(199) : 
    p_b_0[i] = 1/(1 + np.exp(0.5*(m_y[i] - m_b[i])))
    m_b[i+1]= m_b[i] + epsilon*(r_b[i] - m_b[i])
    m_y[i+1] = m_y[i] + epsilon*(r_y[i] - m_y[i])

plt.figure()
plt.plot(x, m_b, 'o', label = "blue flower")
plt.plot(x, m_y, 'o', label = "yellow flower")
plt.xlabel("trial")
plt.ylabel("evolution of the bee's estimate for specific flower rewards")
plt.legend()
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_6_bees.pdf')
plt.show()

#Beta = 0, explorative behaviour
p_b_2 = np.zeros(200)
p_y_2 = np.zeros(200)

for i in range(199) : 
    p_b_2[i] = 1/2
    p_y_2[i] = 1 - p_b_2[i]
    m_b[i+1]= m_b[i] + epsilon*(r_b[i] - m_b[i])
    m_y[i+1] = m_y[i] + epsilon*(r_y[i] - m_y[i])

plt.figure()
plt.plot(x, p_b_2, 'o', label = "blue flower")
plt.xlabel("trial")
plt.ylabel("probability of the bee landing on a specific flower")
plt.legend()
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_7_bees.pdf')
plt.show()

plt.figure()
plt.plot(x, m_b, 'o', label = "blue flower")
plt.plot(x, m_y, 'o', label = "yellow flower")
plt.xlabel("trial")
plt.ylabel("evolution of the bee's estimate for specific flower rewards")
plt.legend()
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_8_bees.pdf')
plt.show()

def plot_smart_bee(beta, m_y_initial, m_b_initial):
    trials = np.zeros(200)
    x = np.linspace(0, 200, 200)
    p_b = np.zeros(200)
    m_y = np.zeros(200)
    m_b = np.zeros(200)
    p_b[0] = 1/(1 + np.exp(beta*(m_y_initial - m_b_initial)))

    for i in range(199) :
        p_b[i] = 1/(1 + np.exp(beta*(m_y[i] - m_b[i])))
        trials[i] = np.random.binomial(1, p_b[i], 1)
        m_b[i+1]= m_b[i] + epsilon*(r_b[i] - m_b[i])
        m_y[i+1] = m_y[i] + epsilon*(r_y[i] - m_y[i])
        

    plt.figure()
    plt.plot(x, trials, 'o', label='1 = blue, 0 = yellow')
    plt.xlabel("trial")
    plt.ylabel("bee's choice blue")
    plt.tick_params("axis='both', labelsize=12")
    plt.savefig('figure_9_bees.pdf')
    plt.show()


#Beta = 1, exploitative behaviour

p_b_3 = np.zeros(200)
p_y_3 = np.zeros(200)

for i in range(199) : 
    p_b_3[i] = 1/(1 + np.exp(m_y[i] - m_b[i]))
    p_y_3[i] = 1 - p_b_3[i]
    m_b[i+1]= m_b[i] + epsilon*(r_b[i] - m_b[i])
    m_y[i+1] = m_y[i] + epsilon*(r_y[i] - m_y[i])

plt.figure()
plt.plot(x, m_b, 'o', label = "m_b")
plt.plot(x, m_y, 'o', label = "m_y")
plt.xlabel("trial")
plt.ylabel("estimated blue flower reward")
plt.legend()
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_10_bees.pdf')
plt.show()


plt.figure()
plt.plot(x, p_b_3, 'o', label = "blue flower")
plt.plot(x, p_y_3, 'o', label = "yellow flower")
plt.xlabel("trial")
plt.ylabel("probability of the bee choosing a flower")
plt.legend()
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_11_bees.pdf')
plt.show()








import numpy as np 
import matplotlib.pyplot as plt

def population_modelV1(p0, time_span, alpha) : 
    #p0 = initial population value, time_span = number of years, alpha = growth factor

    population_array = np.zeros(time_span)
    population_array[0] = p0

    for i in range(1, time_span) :
        population_array[i] = population_array[i-1] + population_array[i-1]*alpha

    return(population_array)


def growth_factor_evolution(population) :
    growth_factor = np.zeros(population)
    growth_factor[0] = 200

    for i in range(1, population) :
        growth_factor[i] = 200 - i

    return(growth_factor)


def population_modelV2(p0, time_span, coeff):
    population_array = np.zeros(time_span)
    population_array[0] = p0

    for i in range(1, time_span) :
        population_array[i] = population_array[i-1] + coeff*population_array[i-1]*(200 - population_array[i-1])

    return(population_array)


Xmax = 100
Npoints = 100
x = np.linspace(0, Xmax, Npoints)

#figure 1
plt.figure()
y1_1 = population_modelV1(2, 100, 0.1)
plt.plot(x, y1_1)
plt.xlabel("time [years]")
plt.ylabel("population")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_1.pdf')
plt.show()


#figure 2
y2_1 = population_modelV1(2, 100, 0.2 )
y2_2 = population_modelV1(2, 100, 0.3)
y2_3 = population_modelV1(2, 100, 0.4)
y2_4 = population_modelV1(2, 100, 0.5)

plt.figure()
plt.plot(x, y2_1, label ="alpha= 0.2")
plt.plot(x, y2_2, label ="alpha = 0.3")
plt.plot(x, y2_3, label ="alpha = 0.4")
plt.plot(x, y2_4, label ="alpha = 0.5")
plt.ylabel("population")
plt.xlabel("time [years]")
plt.yscale("log")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_2.pdf')
plt.show()


#figure 3
y3_1 = population_modelV1(2, 100, 0.1 )
y3_2 = population_modelV1(4, 100, 0.1)
y3_3 = population_modelV1(8, 100, 0.1)
y3_4 = population_modelV1(16, 100, 0.1)

plt.figure()
plt.plot(x, y3_1, label ="N0 = 2")
plt.plot(x, y3_2, label ="N0 = 4")
plt.plot(x, y3_3, label ="N0 = 8")
plt.plot(x, y3_4, label ="N0 = 16")
plt.ylabel("population")
plt.xlabel("time [years]")
plt.yscale("log")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_3.pdf')
plt.show()


#figure 4
Xmax_4 = 500
Npoints_4 = 500

plt.figure()
x4_1 = np.linspace(0, Xmax_4, Npoints_4)
y4_1 = growth_factor_evolution(500)

plt.plot(x4_1, y4_1, label = "growth factor")
plt.xlabel("population")
plt.ylabel("growth factor")
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_4.pdf')
plt.show()


#figure 5
plt.figure()
y5_1 = population_modelV2(2, 100, 0.001)
plt.plot(x, y5_1)
plt.xlabel("time [years]")
plt.ylabel("population")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_5.pdf')
plt.show()


#figure 6

plt.figure()
y6_1 = population_modelV2(2, 100, 0.001)
y6_2 = population_modelV2(2, 100, 0.002)
y6_3 = population_modelV2(2, 100, 0.003)
y6_4 = population_modelV2(2, 100, 0.004)
y6_5 = population_modelV2(2, 100, 0.005)
y6_6 = population_modelV2(2, 100, 0.01)


plt.plot(x, y6_1, label = "coeff = 0.001")
plt.plot(x, y6_2, label = "coeff = 0.002")
plt.plot(x, y6_3, label = "coeff = 0.003")
plt.plot(x, y6_4, label = "coeff = 0.004")
plt.plot(x, y6_5, label = "coeff = 0.005")
plt.plot(x, y6_6, label = "coeff = 0.01")
plt.yscale("log")

plt.xlabel("time [years]")
plt.ylabel("population")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_6.pdf')
plt.show()

#figure 7
y7_1 = population_modelV2(1, 100, 0.001)
y7_2 = population_modelV2(2, 100, 0.001)
y7_3 = population_modelV2(3, 100, 0.001)
y7_4 = population_modelV2(4, 100, 0.001)
y7_5 = population_modelV2(5, 100, 0.001)
y7_6 = population_modelV2(6, 100, 0.001)
y7_7 = population_modelV2(7, 100, 0.001)
y7_8 = population_modelV2(8, 100, 0.001)
y7_9 = population_modelV2(9, 100, 0.001)
y7_10 = population_modelV2(10, 100, 0.001)


plt.plot(x, y7_1, label = "N0 =1")
plt.plot(x, y7_2, label = "N0 =2")
plt.plot(x, y7_3, label = "N0 =3")
plt.plot(x, y7_4, label = "N0 =4")
plt.plot(x, y7_5, label = "N0 =5")
plt.plot(x, y7_6, label = "N0 =6")
plt.plot(x, y7_7, label = "N0 =7")
plt.plot(x, y7_8, label = "N0 =8")
plt.plot(x, y7_9, label = "N0 =9")
plt.plot(x, y7_10, label = "N0 =10")

plt.xlabel("time [years]")
plt.ylabel("population")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_7.pdf')
plt.show()

#figure 8
plt.figure()

y8_1 = population_modelV2(300, 100, 0.001)
y8_2 = population_modelV2(500, 100, 0.001)
y8_3 = population_modelV2(1000, 100, 0.001)

plt.plot(x, y8_1, label = "Initial population =300")
plt.plot(x, y8_2, label = "Initial population =500")
plt.plot(x, y8_3, label = "Initial population = 1000")
plt.yscale("log")
plt.xlabel("time [years]")
plt.ylabel("population")
plt.legend(fontsize=14)
plt.tick_params("axis='both', labelsize=12")
plt.savefig('figure_8.pdf')
plt.show()

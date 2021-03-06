
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import settings
import binary_string
import objective_function


def flip_k_bit(n, k):
    A =  np.zeros((2**n,2**n), dtype=np.int32)

    for i in range(2**n):
        i_flipped = i ^ (2**k)
        A[i,i_flipped] = 1

    return A


def generate_B(n):
    B = np.zeros((2**n,2**n),dtype = np.int32)
    for i in range(n):
        B += flip_k_bit(n,i)

    return B


def C_z(results_array):

    results = []
    for i in range(len(results_array[0])):
        total = 0
        for j in range(len(results_array)):
            total += results_array[j][i]
        results.append(total)

    matrix =  np.zeros((2**settings.n,2**settings.n), dtype=np.int32);

    for i in range(2**settings.n):
        matrix[i][i] = results[i]

    return matrix


def U_b_beta(B, beta):
    return expm(-1j*beta*B)


def U_c_gamma(C, gamma):
    return expm(-1j*C*gamma)



def inner_function(n, results_array, gamma_array, beta_array):

    qstate = np.identity(2**n,dtype= np.complex)
    B  = generate_B(n)
    C = C_z(results_array)
    S = np.array(np.ones(2**n)/np.sqrt(2**n))[np.newaxis]


    for i in range(len(beta_array)):
        beta = beta_array[i]
        gamma = gamma_array[i]
        qstate = qstate.dot(U_b_beta(B,beta))
        qstate = qstate.dot(U_c_gamma(C,gamma))



    qstate = qstate.dot(S.T)

    temp = np.conj(qstate)
    temp = temp.T.dot(C)
    temp = temp.dot(qstate)

    temp = temp.real

    return -1 * temp


def F_p(gamma_array, beta_array):

    results_array = objective_function.gen_results_arr()

    return inner_function(settings.n,results_array,gamma_array,beta_array)

def F_p_test(conc_array):

    half = len(conc_array)//2

    gamma_array = conc_array[:half]
    beta_array = conc_array[half:]

    results_array = objective_function.gen_results_arr()

    return inner_function(settings.n,results_array,gamma_array,beta_array)

def F_p_evolution(n, p_max):
    settings.n = n
    arr = [None] * settings.n
    binary_string.binary_strings(settings.n, arr, 0)
    binary_string.fix_strings()
    results = objective_function.gen_results_strings()
    results_array = objective_function.gen_results_arr()
    p = 2
    p_array = []
    results = []
    function_calls = []

    for i in range(p_max):

        x0 = [np.pi/3]*p
        x1 = [np.pi/4]*p
        x = x0 + x1
        bnds1 = [(0,np.pi)]*(p)
        bnds2 = [(0,2*np.pi)]*(p)
        bnds = bnds1 + bnds2

        sol = minimize(F_p_test,x,method = 'SLSQP',bounds = bnds)

        results.append(sol.fun)
        function_calls.append(sol.njev)
        p_array.append(p)
        p += 1

        # x = []
        # for i in sol.x:
        #     x.append(i)
        # x.append(0)
        # x.append(0)


    return p_array, results, function_calls



    return p_array, results

def H(t, B, C):

    T = 4000


    H_t = B*(1-t/T) + C*(t/T)

    return H_t


def H_optimize(n):
    results_array = objective_function.gen_results_arr()

    T = 4000
    t = np.linspace(0,T,T)
    C = C_z(results_array)
    B = generate_B(n)

    # print(C)
    vec = (np.array(np.ones(2**n)/np.sqrt(2**n))[np.newaxis]).T

    for time in t:
        U = expm(-1j*H(time,B,C))
        vec = U.dot(vec)


    abs_val = np.absolute(vec)
    largest_val = 0
    j = -1
    # print(abs_val)

    for i in range(len(abs_val)):
        if abs_val[i] > largest_val:
            largest_val = abs_val[i]
            j = i

    #
    # print(largest_val, j)
    # print("The matrix value is",C[j][j])



    return C[j][j], j


def H_evolution(n):
    settings.n = n
    arr = [None] * settings.n
    binary_string.binary_strings(settings.n, arr, 0)
    binary_string.fix_strings()
    results = objective_function.gen_results_strings()
    results_array = objective_function.gen_results_arr()



    T = 4000
    t = np.linspace(0,T,T)
    C = C_z(results_array)
    B = generate_B(settings.n)

    vec = (np.array(np.ones(2**settings.n)/np.sqrt(2**settings.n))[np.newaxis]).T

    max_element = []

    for time in t:
        U = expm(-1j*H(time,B,C))
        vec = U.dot(vec)
        max_element.append(max(np.absolute(vec)))
        # abs_val = np.absolute(vec)
        # largest_val = 0
        # j = -1
        #
        # for i in range(len(abs_val)):
        #     if abs_val[i] > largest_val:
        #         largest_val = abs_val[i]
        #         j = i
        #
        # max_element.append(C[j][j])

    return t,max_element





def results_to_file():
    f = open("results.txt", "w")

    settings.n = 3

    for i in range(12):
        arr = [None] * settings.n
        binary_string.binary_strings(settings.n, arr, 0)
        binary_string.fix_strings()
        results = objective_function.gen_results_strings()

        test = H_optimize(settings.n)
        f.write("{}\n".format(settings.n))
        f.write("Adiabatic: Index {} | String: {} | Result: {}\n".format(test[1],settings.strings[test[1]], test[0]))

        ind = results.index(max(results))
        f.write("Manually done: Index {} | String: {} |  Result: {}\n".format(ind,settings.strings[ind],results[ind]))

        if test[0] == results[ind]:
            f.write("CORRECT\n")
        else:
            f.write("INCORRECT\n")

        f.write("\n")

        settings.n += 1

    f.close()

    print("Done!")





if __name__ == "__main__":

    settings.n = 7
    # p = 3

    arr = [None] * settings.n
    settings.init()
    binary_string.binary_strings(settings.n, arr, 0)
    binary_string.fix_strings()
    #
    # results = objective_function.gen_results_strings()
    # print(results)
    # test = H_optimize(settings.n)
    # print()
    # print("H optimize index: {} String: {} Result: {}".format(test[1],settings.strings[test[1]], test[0]))
    #
    #
    # ind = results.index(max(results))
    # print("Manually done index: {} String: {} Result: {}".format(ind,settings.strings[ind],results[ind]))

    # x0 = [np.pi/3]*p
    # x1 = [np.pi/4]*p
    # x = x0 + x1
    # bnds1 = [(0,2*np.pi)]*(2*p)
    #
    # sol = minimize(F_p_test,x,method = 'SLSQP',bounds = bnds1)
    # print(sol)
    # print(F_p(sol.x[:p], sol.x[p:]))



    # x = H_evolution(6)
    # plt.scatter(x[0], x[1], color='r')
    # x = H_evolution(8)
    # plt.scatter(x[0], x[1], color='g')
    # x = H_evolution(10)
    # plt.scatter(x[0], x[1],color='b')
    # red = mpatches.Patch(color='red', label='n=6')
    # green = mpatches.Patch(color='red', label='n=8')
    # blue = mpatches.Patch(color='red', label='n=10')
    # plt.legend(handles=[red,green,blue])
    #
    # plt.xlabel("H")
    # plt.ylabel("Optimal Value")
    # plt.show()

    n_1 = 5
    n_2 = 6
    n_3 = 7
    p = 10

    y_1 = F_p_evolution(n_1,p)
    plt.scatter(y_1[0],[-i for i in y_1[1]], color='r')

    y_2 = F_p_evolution(n_2, p)
    plt.scatter(y_2[0],[-i for i in y_2[1]], color='g')

    y_3 = F_p_evolution(n_3, p)
    plt.scatter(y_3[0],[-i for i in y_3[1]], color='b')

    red = mpatches.Patch(color='red', label='n={}'.format(n_1))
    green = mpatches.Patch(color='green', label='n={}'.format(n_2))
    blue = mpatches.Patch(color='blue', label='n={}'.format(n_3))
    plt.legend(handles=[red,green,blue])


    plt.xlabel("p")
    plt.ylabel("Optimal Value")
    plt.ylim([0,15])
    plt.show()


    # y_3 = F_p_evolution(n_3, p)
    # plt.scatter(y_3[0],[-i for i in y_3[1]], color='b')
    y_f = y_3[2];
    repetitions = 9
    for i in range(repetitions):
        y_3 = F_p_evolution(n_3, p)
        for j in range(p):
            y_f[j] += y_3[2][j]


    for i in y_f:
        y = i/repetitions

    print(y_f)
    plt.plot(y_3[0],y_f, color='b')
    plt.xlabel("p")
    plt.ylabel("Number of function calls")

    plt.show()



    # results_to_file()

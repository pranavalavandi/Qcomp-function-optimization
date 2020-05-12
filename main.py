
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

def F_p_evolution():
    p = 1
    p_array = []
    results = []
    for i in range(8):

        x0 = [np.pi/3]*p
        x1 = [np.pi/4]*p
        x = x0 + x1
        bnds1 = [(0,2*np.pi)]*(2*p)
        sol = minimize(F_p_test,x,method = 'SLSQP',bounds = bnds1)

        results.append(sol.fun)
        p_array.append(p)
        p += 1
        # print(sol)
        # print(F_p(sol.x[:p], sol.x[p:]))

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
    results_array = objective_function.gen_results_arr()



    T = 4000
    t = np.linspace(0,T,T)
    C = C_z(results_array)
    B = generate_B(n)

    vec = (np.array(np.ones(2**n)/np.sqrt(2**n))[np.newaxis]).T

    max_element = []

    for time in t:
        U = expm(-1j*H(time,B,C))
        vec = U.dot(vec)
        max_element.append(max(np.absolute(vec)))


    return t,max_element





def results_to_file():
    f = open("results.txt", "w")

    settings.n = 3

    for i in range(3):
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

    settings.n = 3
    # p = 3

    arr = [None] * settings.n
    settings.init()
    binary_string.binary_strings(settings.n, arr, 0)
    binary_string.fix_strings()
    #
    # results = objective_function.gen_results_strings()
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


    # results_to_file()

    # x = H_evolution(settings.n)
    # plt.scatter(x[0], x[1])
    # plt.show()

    y = F_p_evolution()
    plt.scatter(y[0],y[1])
    plt.show()

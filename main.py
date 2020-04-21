
import numpy as np
from scipy.linalg import expm

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

    matrix =  np.zeros((2**n,2**n), dtype=np.int32);

    for i in range(2**n):
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

    return temp


def F_p(gamma_array, beta_array):

    global strings

    results_array = objective_function.gen_results_arr()

    return inner_function(n,results_array,gamma_vector,beta_vector)



if __name__ == "__main__":

    n = 3
    p = 6
    arr = [None] * n
    # strings = []
    settings.init()
    binary_string.binary_strings(n, arr, 0)
    binary_string.fix_strings()

    gamma_vector = np.linspace(0,2*np.pi, p)
    beta_vector = np.linspace(0,np.pi, p)

    # print(settings.strings)
    print(F_p(gamma_vector,beta_vector))

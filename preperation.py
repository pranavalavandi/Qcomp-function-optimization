import numpy as np
from scipy.linalg import expm

def generate_array(arr, n):
    global strings
    x = ''

    for i in range(0, n):
        x += str(arr[i])
    strings.append(x)


# Function to generate all binary strings
def binary_strings(n, arr, i):
    if i == n:
        generate_array(arr, n)
        return
    arr[i] = 0
    binary_strings(n, arr, i + 1)

    arr[i] = 1
    binary_strings(n, arr, i + 1)

def qubit(n):
    x = 1-2*n
    return x

# C_1(z)= string has #2 1's exactly
def first_subclause(strings):
    results = []

    for i in range(len(strings)):
        sum  = 0
        for j in range(len(strings[i])-1):
            sum += qubit(int(strings[i][j]))*qubit(int(strings[i][j+1]))

        results.append(sum)

    return results

#C_2(z) = string has more than 3 pairs of adjacent 0's
def second_subclause(strings):
    results = []

    for i in range(len(strings)):
        sum = 0
        for j in range(len(strings[i])-3):
            sum += qubit(int(strings[i][j]))*qubit(int(strings[i][j+3]))

        results.append(sum)

    return results

def results_matrix(strings):
    results1 = first_subclause(strings)
    results2 = second_subclause(strings)

    results = []
    for i in range(len(results1)):
        results.append(results1[i] + results2[i])

    matrix =  np.zeros((2**n,2**n), dtype=np.int32);
    for i in range(2**n):
        matrix[i][i] = results[i]

    return matrix

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


def unitary_operator(array_results, gamma,n):
#     array results is a 2d array of results from each subclause

    result = np.identity(2**n,dtype= np.csingle)
    matrix =  np.zeros((2**n,2**n), dtype= np.csingle);

    for i in range(len(array_results)):
        for k in range(2**n):
            matrix[i][i] = array_results[i][k]

        result = result.dot(expm(-1j*gamma*matrix))


    return result


def U(B, beta):
    return expm(-1j*beta*B)

def q_state(gamma, B, beta, S, array_results, n):
    qstate = np.identity(2**n,dtype= np.complex)

    for i in beta:
         qstate = qstate.dot(U(B,i))

    for i in gamma:
        qstate = qstate.dot(unitary_operator(array_results, i, n))

    qstate = qstate.dot(S.T)
    print(qstate)
    print(qstate.shape)
    return qstate


def F_p(qstate, C):

    temp = qstate.T.dot(C)
    temp = temp.dot(qstate)

    return temp


# Driver Code
if __name__ == "__main__":

    n = 2
    gamma = np.pi
    beta = np.pi/2
    p = 15

    arr = [None] * n
    strings = []
    binary_strings(n, arr, 0)

    results1 = first_subclause(strings)
    results2 = second_subclause(strings)
    array_results = [results1,results2]

    S = np.array(np.ones(2**n)/np.sqrt(2**n))[np.newaxis]
    gamma_vector = np.linspace(0,2*np.pi, p)
    beta_vector = np.linspace(0,np.pi, p)




    matrix = results_matrix(strings)
    u_operator = unitary_operator(array_results,gamma,n)
    B = generate_B(n)
    qstate = q_state(gamma_vector,B,beta_vector,S,array_results,n)
    output = F_p(qstate, matrix)


#     print(qstate.shape)

    print("Results Matrix\n")
    print(matrix)

    print("B\n")
    print(B)

    print("qstate\n")
    print(qstate)

    print("Unitary operator\n")
    print(u_operator)

    print("output\n")
    print(output.shape)
    print(output)

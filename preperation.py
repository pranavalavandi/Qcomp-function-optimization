import numpy as np
from scipy.linalg import expm
def generate_array(arr, n):
    global strings
    x = ''
    for i in range(0, n):
#         print(arr[i], end = " ")
        x += str(arr[i])
#         print(x)
    strings.append(x)
#     print()

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
    B =  np.zeros((2**n,2**n), dtype=np.int32)
    for i in range(2**n):
        i_flipped = i ^ (2**k)
        B[i,i_flipped] = 1


    return B

def generate_B(n)
    B = np.zeros((2**n,2**n),dtype = np.int32)
    for i in range(1,n+1):
        B += flip_k_bit(n,i)


    return B


def unitary_operator(array_results, gamma,n):
    # array results is a 2d array of results from each subclause
    result = np.identity(2**n,dtype= np.complex)

    matrix =  np.zeros((2**n,2**n), dtype= np.complex);

    for i in range(len(array_results)):
        for k in range(2**n):
            matrix[i][i] = array_results[i][k]

        result *= expm(-1j*gamma*matrix)


    return result



# Driver Code
if __name__ == "__main__":

    n = 4
    gamma = np.pi
    arr = [None] * n
    strings = []
    binary_strings(n, arr, 0)

#     for i in range(len(strings)):
#         print(strings[i])
#     results1 = first_subclause(strings)
#     results2 = second_subclause(strings)

    results1 = first_subclause(strings)
    results2 = second_subclause(strings)
    array_results = [results1,results2]

    matrix = results_matrix(strings)
    u_operator = unitary_operator(array_results,gamma,n)
    print("Results Matrix\n")
    print(matrix)
    print("Sum of single bit operators")
    print(generate_B(2,0))
    print("Unitary operator")
    print(u_operator)

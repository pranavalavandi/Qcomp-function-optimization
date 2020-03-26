import numpy as np

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

def generate_B(n):
    B =  np.zeros((n,n), dtype=np.int32)

    for i in range(n):
        temp_matrix = np.zeros((n,n),dtype = np.int32)
        for k in range(n):
            if k == i:
                temp_matrix[k][k] = -1
            else:
                temp_matrix[k][k] = 1

        B += temp_matrix

    return B

def generate_B_2(n):
    matrix = np.zeros((n,n), dtype=np.int32)
    for i in range(n):
        matrix[i][i] = n - 2

    return matrix

def unitary_operator(C_one,C_two, gamma,n):
    result = np.identity(2**n,dtype= float)

    matrix =  np.zeros((2**n,2**n), dtype=float);
    for i in range(2**n):
        matrix[i][i] =  np.exp(-1j * gamma * C_one[i])

    result = result * matrix


    matrix =  np.zeros((2**n,2**n), dtype=float);
    for i in range(2**n):
        matrix[i][i] = np.exp(-1j * gamma *C_two[i])

    result = result * matrix




    #
    # for i in range(2**n):
    #     C_one[i][i] = np.exp(-1j*gamma*(C_one[i][i]))
    #
    # for i in range(2**n):
    #     C_two[i][i] = np.exp(-1j*gamma*(C_two[i][i]))


#     result = result * C_1 * C_2

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

#     results = []
#     for i in range(len(results1)):
#         results.append(results1[i] + results2[i])

#     matrix =  np.zeros((2**n,2**n), dtype=np.int32);
#     for i in range(2**n):
#         matrix[i][i] = results[i]

    results1 = first_subclause(strings)
    results2 = second_subclause(strings)

    matrix = results_matrix(strings)
    u_operator = unitary_operator(results1,results2,gamma,n)
    print("Results Matrix\n")
    print(matrix)
    print("Sum of single bit operators")
    print(generate_B_2(n))
    print("Unitary operator")
    print(u_operator)

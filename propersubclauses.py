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


def first_subclause(strings):
    results = []
    for i in range(len(strings)):
        sum  = 0
        for j in range(len(strings[i])-1):
            sum += qubit(int(strings[i][j]))*qubit(int(strings[i][j+1]))
        results.append(sum)

def second_subclause(strings):
    results = []
    for i in range(len(strings)):
        sum = 0
        for j in range(len(strings[i])-3):
            sum += qubit(int(strings[i][j]))*qubit(int(strings[i][j+3]))

        results.append(sum)

    return results

if __name__ == "__main__":

    n = 4
    arr = [None] * n
    strings = []

    binary_strings(n, arr, 0)
    results1 = first_subclause(strings)
    results2 = second_subclause(strings)
    results = []
    for i in range(len(results1)):
        results.append(results1[i] + results2[i])

    matrix =  np.zeros((2**n,2**n), dtype=np.int32);
    for i in range(2**n):
        matrix[i][i] = results[i]

    print(matrix)

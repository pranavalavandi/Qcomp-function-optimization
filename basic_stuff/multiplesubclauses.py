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


# C_1(z)= string has #2 1's exactly
def first_subclause(strings):
    results = []
    count = 0
    for i in range(len(strings)):
        count = 0
        for l in range(len(strings[i])):
            if strings[i][l] == "1":
                count += 1
        if(count == 2):
            results.append(1)
        else:
            results.append(0)
    return results


#C_2(z) = string has more than 3 pairs of adjacent 0's
def second_subclause(strings):
    results = []
    count = 0
    for i in range(len(strings)):
        count = 0
        for l in range(len(strings[i]) - 1):
            if strings[i][l] == "0" and strings[i][l+1] == '0':
                count += 1
        if(count >= 3):
            results.append(1)
        else:
            results.append(0)
    return results





# Driver Code
if __name__ == "__main__":

    n = 6
    arr = [None] * n
    strings = []

    binary_strings(n, arr, 0)
    print("testing")
    for i in range(len(strings)):
        print(strings[i])
    results1 = first_subclause(strings)
    results2 = second_subclause(strings)
    results = []
    for i in range(len(results1)):
        results.append(results1[i] + results2[i])

    matrix =  np.zeros((2**n,2**n), dtype= np.int32);
    for i in range(2**n):
        matrix[i][i] = results[i]

    print(matrix)

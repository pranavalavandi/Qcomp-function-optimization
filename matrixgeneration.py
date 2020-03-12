
def generate_array(arr, n):
    global strings
    x = ''
    for i in range(0, n):
#         print(arr[i], end = " ")
        x += str(arr[i])
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

#define the objective function to maximise the number of 1's in an n bit binary string

def objective_function(strings):
    results = []
    count = 0
    for i in range(len(strings)):
        for l in range(len(strings[i])):
            if strings[i][l] == 1:
                count += 1
        results.append(count)

# Driver Code
if __name__ == "__main__":

    n = 4
    arr = [None] * n
    strings = []

    binary_strings(n, arr, 0)
    print("testing")
    for i in range(len(strings)):
        print(results[i])

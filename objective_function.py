import settings
import networkx as nx

def qubit(b):
    return 1-2*b

def subclause_1(strings):
    results = []

    for i in range(len(strings)):
        sum  = 0
        for j in range(len(strings[i])-1):
            sum += qubit(strings[i][j])*qubit(strings[i][j+1])

        results.append(sum)

    for i in range(len(strings)):
        sum  = 0
        for j in range(len(strings[i])-2):
            sum += -qubit(strings[i][j])*qubit(strings[i][j+2])

        results[i] += sum

    return results


def subclause_2(strings):
    results = []

    for i in range(len(strings)):
        sum  = 0
        for j in range(len(strings[i])-2):
            sum += qubit(strings[i][j])^qubit(strings[i][j+1])^qubit(strings[i][j+2])

        results.append(sum)

    return results

def subclause_3(strings):
    results = []

    for i in range(len(strings)):

        sum  = 0
        if strings[i][0] == 1 and strings[i][-1] == 1:
            results.append(1)
        else:
            results.append(0)

    return results
    

def gen_results_arr():
    # global strings
    results = []
    x = settings.strings
    results.append(subclause_1(x))
    results.append(subclause_2(x))
    results.append(subclause_3(x))

    return results



def gen_results_strings():
    results = []
    results_array = gen_results_arr()
    for i in range(len(results_array[0])):
        total = 0
        for j in range(len(results_array)):
            total += results_array[j][i]
        results.append(total)

    return results

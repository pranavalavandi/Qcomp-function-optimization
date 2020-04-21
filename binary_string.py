import settings


def generate_array(arr, n):
    # global strings
    x = ''

    for i in range(0, n):
        x += str(arr[i])
    settings.strings.append(x)

def binary_strings(n, arr, i):
    if i == n:
        generate_array(arr, n)
        return
    arr[i] = 0
    binary_strings(n, arr, i + 1)

    arr[i] = 1
    binary_strings(n, arr, i + 1)


def fix_strings():
    # global strings
    settings.strings = [list(string) for string in settings.strings]

    for i in range(len(settings.strings)):
        for j in range(len(settings.strings[i])):
            settings.strings[i][j] = int(settings.strings[i][j])

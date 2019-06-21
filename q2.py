import numpy as np
from time import time
import matplotlib.pyplot as plt
import math


# question 2, part 1
def my_mul3(A):
    result2 = np.zeros(A.shape)
    result3 = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[1]):
                result2[i][j] += A[i][k] * A[k][j]
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[1]):
                result3[i][j] += A[i][k] * result2[k][j]
    return result3


# question 2, part 2
def np_mul3(A):
    return np.matmul(A, np.matmul(A, A))


# question 2, part 3
def compare(n):
    A = np.random.randn(n, n)
    start = time()
    A2 = my_mul3(A)
    end = time()

    start2 = time()
    A2 = np_mul3(A)
    end2 = time()

    return [end - start, end2 - start2]


# question 2, part 4
def show_graph():
    used_time1 = []
    used_time2 = []
    for i in range(1, 101, 10):
        runtime = compare(i)
        used_time1.append(runtime[0])
        used_time2.append(runtime[1])
    x = range(1, 101, 10)
    # print(used_time2)
    plt.plot(x, used_time1, linestyle='-', marker='x', color='r')
    plt.plot(x, used_time2, linestyle='--', marker='o', color='b')
    plt.legend(('my_mul3_time', 'np_mul3_time'))
    plt.ylim((-1, 5))
    plt.title('Vectorization')
    plt.ylabel('Time(s)')
    plt.xlabel('Matrix size')
    plt.grid(True)
    plt.show()


show_graph()

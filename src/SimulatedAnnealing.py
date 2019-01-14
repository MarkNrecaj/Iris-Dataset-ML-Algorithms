from StateNode import StateNode
from heapq import heappush, heappop
import numpy as np
import random
import math

def accept_move(t, delta):
    if delta < 0:
        return True
    if delta == 0:
        return False
    if random.random() < math.exp((-1 * delta)/t):
        return True
    return False

def do_simulated_annealing(dataset, k, r_ratio, alpha, t_start, t_iter):
    t = t_start
    node = StateNode(dataset, k)
    while(t > 0.1):
        for i in range(t_iter):
            current_score = node.get_score()
            neighbours = node.get_children()
            random.shuffle(neighbours)

            print(str(t) + ", " + str(current_score))
            for candidate in neighbours:
                delta = candidate.get_score() - current_score
                if accept_move(t, delta):
                    node = candidate
                    break

        t = alpha * t
    return node



dataset = np.loadtxt('../data/iris.txt')

# Algorithm parameters
r_ratio = 0.01
k = 4
alpha = 0.95
t_start = 10
t_iter = 100

print('r_ratio: ' + str(r_ratio))
print('k: ' + str(k))
print('alpha: ' + str(alpha))
print('t_start: ' + str(t_start))
print('t_iter: ' + str(t_iter))
print('========================')

do_simulated_annealing(dataset, k, r_ratio, alpha, t_start, t_iter)
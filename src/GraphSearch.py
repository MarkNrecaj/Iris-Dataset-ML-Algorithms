from StateNode import StateNode
from heapq import heappush, heappop
import numpy as np
from timeit import default_timer as timer
import csv

dataset = np.loadtxt('../data/iris.txt')
outfile = open('../out/iris_k4.csv', 'w')
writer = csv.writer(outfile, lineterminator='\n')

for i in range(100):
    initial_node = StateNode(dataset, k=3)

    # Algorithm parameters
    r_ratio = 0.01
    max_iter = 10000

    # Uniform cost graph search
    pqueue = []
    heappush(pqueue, initial_node)
    i = 0
    start = timer()
    prev_score = initial_node.get_score()
    while i < max_iter:
        expand_node = heappop(pqueue)
        current_score = expand_node.get_score()
        if prev_score <= current_score and i > 0:
            end = timer()
            print(str([current_score, end-start, i]))
            writer.writerow([current_score, end-start, i])
            break
        # print(str(i) + '. Score: ' + str(current_score))
        prev_score = current_score
        for child in expand_node.get_children():
            heappush(pqueue, child)
            i += 1

outfile.close()

from StateNode import StateNode
from heapq import heappush, heappop
import numpy as np

def is_tabu(tabu_moves, node):
    center_moved = node.center_moved
    direction_moved = node.direction_moved

    if center_moved < 0 or direction_moved < 0: # node is the initial node
        return False

    return tabu_moves[center_moved, direction_moved] > 0


def do_tabu_search(dataset, k, r_ratio, max_iter, tabu_tenure):
    current_node = StateNode(dataset, k)
    tabu_moves = np.zeros((current_node.k, 2 * current_node.d))
    best_state = current_node

    for iter in range(max_iter):
        positive_i = np.where(tabu_moves > 0)
        tabu_moves[positive_i] -= 1

        neighbours = current_node.get_children()
        neighbours_pq = []
        for neighbour in neighbours:
            heappush(neighbours_pq, neighbour)

        best_candidate = None
        while best_candidate is None:
            if len(neighbours_pq) == 0:
                return best_state
            neighbour = heappop(neighbours_pq)
            if not is_tabu(tabu_moves, neighbour):
                best_candidate = neighbour

            elif neighbour < best_state:
                # aspiration by default
                best_candidate = neighbour
                best_state = best_candidate

        current_node = best_candidate
        reverse_index = best_candidate.direction_moved + (1 - 2 *(best_candidate.direction_moved % 2))
        tabu_moves[best_candidate.center_moved, reverse_index] = tabu_tenure
        #print(str(iter) + '. ' + str(current_node.get_score()))

    return best_state

dataset = np.loadtxt('../data/iris.txt')

# Algorithm parameters
r_ratio = 0.01
max_iter = 500
tabu_tenure = 15
k = 4

print('r_ratio: ' + str(r_ratio))
print('max_iter: ' + str(max_iter))
print('tabu_tenure: ' + str(tabu_tenure))
print('k:' + str(k))
print('========================')

for i in range(100):
    result = do_tabu_search(dataset, k, r_ratio, max_iter, tabu_tenure)
    print(str(i) + str(result.get_score()) + str(np.sort(result.centers)))


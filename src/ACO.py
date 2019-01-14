from StateNode import StateNode
import numpy as np
from numpy.random import choice


class Ant:
    def __init__(self, state_node, r_ratio):
        self.state = state_node
        self.__path = []
        self.__path.append(state_node)
        self.__best_state = state_node
        self.r_ratio = r_ratio

    def get_best(self):
        return self.__best_state

    # steps the ant forwards until iter_max iterations has passed w/o
    # improving the best score it's seen
    def forwards(self, pheromone_matrix, num_steps_max):
        i_since_improve = 0
        while i_since_improve < num_steps_max:
            next_node = self.__step(pheromone_matrix)
            if next_node.get_score() >= self.__best_state.get_score():
                i_since_improve += 1
            else:
                self.__best_state = next_node
                i_since_improve = 0
            self.__path.append(next_node)

    # traverses the ant backwards
    # returns: T_updated, after deploying pheromone
    def backwards(self, pheromone_matrix, Q):
        # Remove moves at end which don't improve score
        self.__path = self.__path[0:self.__path.index(self.__best_state) + 1]
        # Detect and remove cycles in the path
        self.__path = self.__detectAndRemoveLoop()
        return self.__apply_pheromone(pheromone_matrix, Q)

    # probabilistically chooses a neighbour based on pheromone levels
    # returns: selected neighbour node
    def __step(self, T):
        children = self.__path[-1].get_children()

        # Prevent the ant from going backwards
        for i in range(len(children) - 1, -1, -1):
            child = children[i]
            if len(self.__path) > 1 and self.__path[-2] == child:
                del children[i]

        prob = self.__get_p(children, T)
        next_node = choice(children, 1, prob)[0]
        return next_node

    # Gets normalized probability vector for a list of StateNodes
    def __get_p(self, children, T):
        probs = []
        for child in children:
            probs.append(self.__get_t(child, T))
        probs_norm = [prob / sum(probs) for prob in probs]
        return probs_norm

    # Get the pheromone level for a node from T
    def __get_t(self, node, pheromone_matrix):
        t_total = 0.0
        for k_i in range(node.k):
            center = node.centers[k_i]
            indices = self.__get_indices(center)
            t_total += pheromone_matrix[tuple(indices)]
        return t_total

    # Given list of StateNodes, treat it as a path and return a new list without cycles
    def __detectAndRemoveLoop(self):
        r = []
        for node in self.__path:
            if node in r:
                r = r[0:self.__path.index(node)]
            r.append(node)
        return r

    # Based on path apply phermones to T
    # returns: T_updated
    def __apply_pheromone(self, T, delta_t_base):
        score_ratio = self.__path[0].get_score() / self.__best_state.get_score() # start score / best score
        indices_set = set()
        delta_t = delta_t_base * (score_ratio - 1)
        # print('========')
        # print(str(self.__path[0].get_score()))
        # print(str(self.__path[-1].get_score()))
        # print(score_ratio)
        k = self.__best_state.k
        for node in self.__path:
            for i in range(k):
                center = node.centers[i]
                indices = self.__get_indices(center)
                indices_set.add(tuple(indices))

        for indices in indices_set:
            new_score = T[indices] + delta_t
            T[indices] = new_score

        return T

    def __get_indices(self, center):
        indices = []
        node = self.get_best()
        for i in range(node.d):
            bound = node.bounds[i]  # get the bound for the ith dimension
            delta = center[i] - bound[0]
            step_size = node.r_ratio * (bound[1] - bound[0])
            index = int(round(delta / step_size)) - 1
            indices.append(index)
        return tuple(indices)


def do_ACO(dataset, k, r_ratio, num_ants, num_steps_max, evap_rate, t_init, delta_p_base):
    node = StateNode(dataset, k, r_ratio=r_ratio)
    T_shape = [int(1 / r_ratio) for i in range(node.d)]
    pheromone_matrix = np.full(tuple(T_shape), t_init)
    ants = [Ant(StateNode(dataset, k, r_ratio=r_ratio), r_ratio) for i in range(num_ants)]
    # ants = [Ant(node, r_ratio) for i in range(num_ants)]
    best_state = node

    print('Starting ACO:')
    print('Params: num_ants:{}, num_steps_max: {}, evap_rate: {}, t_init: {}, Q: {}'
          .format(str(num_ants), str(num_steps_max), str(evap_rate), str(t_init), str(delta_p_base)))
    print('Initial Score: ' )
    print('============')

    for index, ant in enumerate(ants):
        ant.forwards(pheromone_matrix, num_steps_max)
        pheromone_matrix = ant.backwards(pheromone_matrix, delta_p_base)
        pheromone_matrix = evaporate(pheromone_matrix, evap_rate)

        if ant.get_best().get_score() < best_state.get_score():
            best_state = ant.get_best()
        print(str(index) + '. personal best: ' + str(round(ant.get_best().get_score())) + 'global best: ' + str(round(best_state.get_score())))

    return best_state


def evaporate(pheromone_matrix, evap_rate):
    return pheromone_matrix * (1 - evap_rate)


# Load dataset/problem instance variables
dataset = np.loadtxt('../data/ruspini.txt')
k = 3
r_ratio = 0.05

# Set algo params
num_ants = 100
num_steps_max = 120
evap_rate = 0.05
t_init = 150
delta_p_base = 50

do_ACO(dataset, k, r_ratio, num_ants, num_steps_max, evap_rate, t_init, delta_p_base)

import numpy as np
from random import randint


class StateNode:
    def __init__(self, data, k, center_values = np.zeros(0), center_moved = -1, direction_moved = -1, r_ratio = 0.05):
        self.k = k
        self.d = data.shape[1]
        self.center_moved = center_moved
        self.direction_moved = direction_moved
        self.data = data
        self.r_ratio = r_ratio

        self.bounds = np.zeros((self.d, 2))
        for index, col in enumerate(data.T):
            self.bounds[index, 0] = np.amin(col)
            self.bounds[index, 1] = np.amax(col)

        self.centers = np.zeros((k, self.d))

        if center_values.size > 0:
            self.centers = center_values
        else:
            for index_center in range(k):
                for index, bound in enumerate(self.bounds):
                    rand = randint(0, int(1/r_ratio))
                    step_size = (r_ratio* (bound[1] - bound[0]))
                    self.centers[index_center, index] = bound[0] + (rand * step_size)

        self.__set_score()

    def __set_score(self):
        self.__score = 0.00
        for point in self.data:
            difference = point - self.centers
            distances = np.linalg.norm(difference, axis=1)
            self.__score += np.amin(distances) ** 2

    def get_score(self):
        return self.__score

    # return the list of possible children nodes from this point (2*d + 1)*k
    def get_children(self):
        # generate possible move matrix
        moves = np.zeros((2 * self.d, self.d))
        for index, col in enumerate(moves.T):
            r = self.r_ratio * np.ptp(self.bounds[index])
            moves[2 * index, index] = r
            moves[2 * index + 1, index] = r * -1
        # find all valid movements,
        children = []
        for center_index, center in enumerate(self.centers):
            for move_index, move in enumerate(moves):
                if (self.__move_is_valid(center, move)):
                    new_center_value = np.copy(self.centers)
                    new_center_value[center_index] = center + move
                    child = StateNode(self.data, self.k, new_center_value, center_index, move_index)
                    children.append(child)

        return children

    # helper for getChildren()
    def __move_is_valid(self, center, move):
        new_center = center + move
        for index, bound in enumerate(self.bounds):
            if new_center[index] < bound[0] or new_center[index] > bound[1]:
                return False
        return True

    def __lt__(self, other):
        return self.__score < other.__score

    def __eq__(self, other):
        if self.d == other.d and self.k == other.k:
            for center_pair in zip(self.centers, other.centers):
                if not np.all(center_pair[0] == center_pair[1]):
                    return False
            return True
        return False

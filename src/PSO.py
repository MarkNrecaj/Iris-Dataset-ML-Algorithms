import math
import numpy as np
import random as randint

class Particle:
    def __init__(self, data, k, weight, c_cog, c_soc):
        self.k = k
        self.d = data.shape[1]
        self.weight = weight
        self.c_cog = c_cog
        self.c_soc = c_soc
        self.velocities = np.zeros((self.k, self.d))
        self.best_centers = np.zeros((self.k, self.d))

        # max min values for every dimension
        self.bounds = np.zeros((self.d, 2))
        for index, col in enumerate(data.T):
            self.bounds[index, 0] = np.amin(col)
            self.bounds[index, 1] = np.amax(col)

        # k rows of d dimensions, every row is k cluster
        self.centers = np.zeros((self.k, self.d))

        #iterate over k (rows)
        for index_center in range(k):
            #iterate cols
            for index, bound in enumerate(self.bounds):
                rand = np.random.uniform(bound[0], bound[1])
                self.centers[index_center, index] = rand

        self.best_centers = self.centers
        self.__set_score(data)

    def move(self, global_best_centers, data):
        # particle moves closer to solution based on previous data and momentum
        # enumerate all possibilities for closest two particles
        # TODO: map centers to each other, using indexing
        # Updates velocity matrix based on inertia, cognitive and social parameters
        for i, velocity in enumerate(self.velocities):
            inertia = self.weight*self.velocities[i]
            cog = self.c_cog*np.random.uniform(0, 1)*(self.best_centers[i]-self.centers[i])
            soc = self.c_soc*np.random.uniform(0, 1)*(global_best_centers[i]-self.centers[i])
            self.velocities[i] = inertia+cog+soc
        prev_score = self.get_score(data)
        # Move the cluster centers based on velocities
        for i, center in enumerate(self.centers):
            self.centers[i] = center + self.velocities[i]
        self.__set_score(data)
        if prev_score > self.get_score(data):
            self.best_centers = self.centers

    def __set_score(self, data):
        self.__score = 0.00
        for point in data:
            difference = point - self.centers
            distances = np.linalg.norm(difference, axis=1)
            self.__score += np.amin(distances) ** 2

    def get_score(self, data):
        self.__set_score(data)
        return self.__score


def do_PSO(data, k, weight, c_cog, c_soc, num_particles, max_iter):
    particles = []
    global_best_centers = np.zeros((k,data.shape[1]))
    best_score = float('Inf')
    # Worst score in infinity, get score and see if its lower then previous best (initially INF)
    for i in range(num_particles):
        particle = Particle(data, k, weight, c_cog, c_soc)
        particles.append(particle)
        if particle.get_score(data) < best_score:
            best_score = particle.get_score(data)
            global_best_centers = particle.centers
    # move global best towards roost based on scores
    for i in range(max_iter):
        for particle in particles:
            particle.move(global_best_centers, data)
            if particle.get_score(data) < best_score:
                best_score = particle.get_score(data)
                global_best_centers = particle.centers

                print(best_score)
        #print(best_score)
# w = 0.3, c_cog = 0.3, c_soc = 0.7
data = np.loadtxt('../data/iris.txt')
#CHANGE VALUES HERE
k=2; weight=0.2; c_cog=0.7; c_soc=1.5; num_particles=1000; max_iter=100000
print('k:', k)
print('weight:', weight)
print('c_cog:', c_cog)
print('c_soc:', c_soc)
print('num_particles:', num_particles)
do_PSO(data, k, weight, c_cog, c_soc, num_particles, max_iter)
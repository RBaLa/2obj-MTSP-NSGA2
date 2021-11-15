import numpy as np
import matplotlib.pyplot as plt
import math
import copy

def objective_function_1(C,X):
    n_salespersons = X.shape[0]
    n_cities = C.shape[0]
    total_distance = 0
    
    for i,tour in enumerate(X):
        distance = np.sum(np.multiply(C,tour))
        total_distance += distance
        
    return total_distance

def objective_function_2(C,X,ftype='td'):
    n_salespersons = X.shape[0]
    n_cities = C.shape[0]
    tour_lengths = np.empty(shape=(n_salespersons,),dtype=float)
    total_distance = 0
    avg_tour_length = 0
    variation = 0
    
    if ftype=='td': #Total deviation from average
        for i,tour in enumerate(X):
            distance = np.sum(np.multiply(C,tour))
            total_distance += distance
            tour_lengths[i] = distance
        avg_tour_length = total_distance/n_salespersons
        for i in range(n_salespersons):
            variation += np.abs(avg_tour_length - tour_lengths[i])
        return variation
    
    if ftype=='r': #Range of tour lengths
        for i,tour in enumerate(X):
            tour_lengths[i] = np.sum(np.multiply(C,tour))
        return max(tour_lengths)-min(tour_lengths)
    
    if ftype=='std': #Standard deviation
        for i,tour in enumerate(X):
            distance = np.sum(np.multiply(C,tour))
            total_distance += distance
            tour_lengths[i] = distance
        avg_tour_length = total_distance/n_salespersons
        for i in range(n_salespersons):
            variation += (avg_tour_length - tour_lengths[i])**2
        return (variation/(n_salespersons-1))**0.5

def get_tour_matrix(chromosome,ctype=0):
    if ctype==0:
        ntours = max(chromosome.salespersons) #same as no. of salespersons
        ncities = max(chromosome.cities)
        tour_matrix = np.zeros(shape=(ntours,ncities+1,ncities+1),dtype=int)
        tour_order_lists = [[] for i in range(ntours)]
        for i in range(ncities):
            tour_order_lists[chromosome.salespersons[i]-1].append(chromosome.cities[i])
        
    if ctype==1:
        part_1 = chromosome.part_1
        part_2 = chromosome.part_2
        ntours = part_2.shape[0]+1 #again, this is no. of salespersons
        ncities = part_1.shape[0]
        tour_matrix = np.zeros(shape=(ntours,ncities+1,ncities+1),dtype=int)
        tour_order_lists = [[] for i in range(ntours)]
        count = 0
        breakpoints = [part_2[i] for i in range(ntours-1)]
        breakpoints.append(ncities)
        for i in range(ntours):
            breakpoint = breakpoints[i]
            for j in range(count,breakpoint):
                tour_order_lists[i].append(part_1[j])
            count = copy.deepcopy(breakpoint)
    
    for i,tours in enumerate(tour_order_lists):
        for j,city in enumerate(tours):
            if j!=0:
                tour_matrix[i,tours[j-1],city] = 1
            else:
                tour_matrix[i,0,city] = 1
            if j==len(tours):
                tour_matrix[i,city,0] = 1
                
    return tour_matrix

class Chromosome_1(object):
    def __init__(self):
        self.salespersons = None
        self.cities = None
    def __eq__(self,other):
        if isinstance(self,other.__class__):
            return ((np.all(self.salespersons==other.salespersons)) and (np.all(self.cities==other.cities)))
        return False

class Chromosome_2(object):
    def __init__(self):
        self.part_1 = None
        self.part_2 = None
    def __eq__(self,other):
        if isinstance(self,other.__class__):
            return ((self.salespersons==other.salespersons) and (self.cities==other.cities))
        return False

def create_chromosome_1(n_cities,n_salespersons):
    while True:
        check = True
        chromosome = Chromosome_1()
        rng = np.random.default_rng()
        chromosome.cities = rng.permutation(np.arange(1,n_cities))
        chromosome.salespersons = rng.choice(np.arange(1,n_salespersons+1),n_cities-1)
        for i in range(1,n_salespersons+1):
            if i not in chromosome.salespersons:
                check = False
        if check == True:    
            return chromosome

def create_chromosome_2(n_cities,n_salespersons):
    chromosome = Chromosome_2()
    rng = np.random.default_rng()
    chromosome.part_1 = rng.permutation(np.arange(1,n_cities))
    chromosome.part_2 = np.sort(rng.choice(np.arange(1,n_cities-1),n_salespersons-1,replace=False))
    return chromosome

def euclidean_distance(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

def generate_instance(n_cities,map_size):
    rng = np.random.default_rng()
    city_coordinates = [(0,0)]
    distance_matrix = np.empty(shape=(n_cities,n_cities),dtype=float)
    while len(city_coordinates)<n_cities:
        x_choice = rng.choice(map_size)
        y_choice = rng.choice(map_size)
        new_city = (x_choice,y_choice)
        if new_city not in city_coordinates:
            city_coordinates.append(new_city)
    for i,city_i in enumerate(city_coordinates):
        for j,city_j in enumerate(city_coordinates):
            distance_matrix[i,j] = euclidean_distance(city_i,city_j)
    plt.figure(figsize=(8,8))
    x = [p[0] for p in city_coordinates]
    y = [p[1] for p in city_coordinates]
    plt.xlim([-map_size/100,map_size+map_size/100])
    plt.ylim([-map_size/100,map_size+map_size/100])
    plt.plot(x,y,"ob")
    plt.plot(0,0,"or")
    plt.grid()
    plt.show()
    
    return distance_matrix

def create_initial_population_1(N,n_cities,n_salespersons):
    population = []
    while len(population)<N:
        individual = create_chromosome_1(n_cities,n_salespersons)
        if individual not in population:
            population.append(individual)
    return population

def create_initial_population_2(N,n_cities,n_salespersons):
    population = []
    while len(population)<N:
        individual = create_chromosome_2(n_cities,n_salespersons)
        if individual not in population:
            population.append(individual)
    return population
    
if __name__=="__main__":
    pass

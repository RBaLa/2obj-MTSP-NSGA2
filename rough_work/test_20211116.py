import numpy as np
import matplotlib.pyplot as plt
import math
import copy

SEED = 123456789

def updateSeed():
    global SEED
    large = 2147483647;
    k = int(SEED/127773);
    SEED = 16807*(SEED-k*127773)-k*2836;
    if SEED<=0:
        SEED += large;

def objectiveFunction1(C,X):
    n_salespersons = X.shape[0]
    n_cities = C.shape[0]
    total_distance = 0
    for i,tour in enumerate(X):
        distance = np.sum(np.multiply(C,tour))
        total_distance += distance
    return total_distance

def objectiveFunction2(C,X,ftype='td'):
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

def getTourMatrix(chromosome,ctype=1):
    if ctype==1:
        n_tours = max(chromosome.salespersons) #same as no. of salespersons
        n_cities = max(chromosome.cities)
        tour_matrix = np.zeros(shape=(n_tours,n_cities+1,n_cities+1),dtype=int)
        tour_order_lists = [[] for i in range(n_tours)]
        for i in range(n_cities):
            tour_order_lists[chromosome.salespersons[i]-1].append(chromosome.cities[i])
        
    if ctype==2:
        part_1 = chromosome.part_1
        part_2 = chromosome.part_2
        n_tours = part_2.shape[0]+1 #again, this is no. of salespersons
        n_cities = part_1.shape[0]
        tour_matrix = np.zeros(shape=(n_tours,n_cities+1,n_cities+1),dtype=int)
        tour_order_lists = [[] for i in range(n_tours)]
        count = 0
        breakpoints = [part_2[i] for i in range(n_tours-1)]
        breakpoints.append(n_cities)
        for i in range(n_tours):
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
            return ((np.all(self.part_1==other.part_1)) and (np.all(self.part_2==other.part_2)))
        return False

def createChromosome(n_cities,n_salespersons,ctype=1):
    if ctype==1:
        while True:
            check = True
            chromosome = Chromosome_1()
            rng = np.random.default_rng(SEED)
            updateSeed()
            chromosome.cities = rng.permutation(np.arange(1,n_cities))
            chromosome.salespersons = rng.choice(np.arange(1,n_salespersons+1),n_cities-1)
            for i in range(1,n_salespersons+1):
                if i not in chromosome.salespersons:
                    check = False
            if check == True:    
                return chromosome
    if ctype==2:
        chromosome = Chromosome_2()
        rng = np.random.default_rng(SEED)
        updateSeed()
        chromosome.part_1 = rng.permutation(np.arange(1,n_cities))
        chromosome.part_2 = np.sort(rng.choice(np.arange(1,n_cities-1),n_salespersons-1,replace=False))
        return chromosome

def euclideanDistance(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

def generateInstance(n_cities,map_size):
    rng = np.random.default_rng(SEED)
    updateSeed()
    city_coordinates = [(0,0)] #initialized with depot
    distance_matrix = np.empty(shape=(n_cities,n_cities),dtype=float)
    while len(city_coordinates)<n_cities:
        x_choice = rng.choice(map_size)
        y_choice = rng.choice(map_size)
        new_city = (x_choice,y_choice)
        if new_city not in city_coordinates: #ensuring only new entries
            city_coordinates.append(new_city)
    for i,city_i in enumerate(city_coordinates):
        for j,city_j in enumerate(city_coordinates):
            distance_matrix[i,j] = euclideanDistance(city_i,city_j)
    #Plot generate instance:
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

def createInitialPopulation(N,n_cities,n_salespersons,ctype=1):
    population = []
    while len(population)<N:
        individual = createChromosome(n_cities,n_salespersons,ctype)
        if individual not in population:
            population.append(individual)
    return population

def partiallyMappedCrossover(p1,p2,ctype=1):
    rng = np.random.default_rng(SEED)
    updateSeed()
    if ctype==1:
        c_1,s_1 = p1.cities,p1.salespersons
        c_2,s_2 = p2.cities,p2.salespersons
        child_1 = Chromosome_1()
        child_1.cities = np.zeros(shape=c_1.shape,dtype=int)
        child_1.salespersons = np.zeros(shape=s_1.shape,dtype=int)
        child_2 = Chromosome_1()
        child_2.cities = np.zeros(shape=c_2.shape,dtype=int)
        child_2.salespersons = np.empty(shape=s_2.shape,dtype=int)
        cut_points = np.sort(rng.choice(c_1.shape[0],2,replace=False))
        for i in range(cut_points[0],cut_points[1]):
            child_2.cities[i] = c_1[i]
            child_2.salespersons[i] = s_1[i]
            child_1.cities[i] = c_2[i]
            child_1.salespersons[i] = s_2[i]
        for i in range(c_1.shape[0]):
            if c_1[i] not in child_1.cities:
                child_1.cities[i] = c_1[i]
                child_1.salespersons[i] = s_1[i]
            else:
                child_1.cities[i] = rng.choice([j for j in range(1,c_1.shape[0]+1) if j not in child_1.cities])
                if np.any(s_1) not in child_1.salespersons:
                    child_1.salespersons[i] = (rng.choice(j) for j in s_1 if j not in child_1.salespersons)
                else:
                    child_1.salespersons[i] = rng.choice(np.arange(1,max(s_1)+1))
            if c_2[i] not in child_2.cities:
                child_2.cities[i] = c_2[i]
                child_2.salespersons[i] = s_2[i]
            else:
                child_2.cities[i] = rng.choice([j for j in range(1,c_2.shape[0]+1) if j not in child_2.cities])
                if np.any(s_2) not in child_2.salespersons:
                    child_2.salespersons[i] = (rng.choice(j) for j in s_2 if j not in child_2.salespersons)
                else:
                    child_2.salespersons[i] = rng.choice(np.arange(1,max(s_2)+1))
        
    if ctype==2:
        child_1 = Chromosome_2()
        child_2 = Chromosome_2()
        p11 = p1.part_1
        p21 = p2.part_1
        cut_points = np.sort(rng.choice(p11.shape[0],2,replace=False))
        for i in range(cut_points[0],cut_points[1]):
            child_2.part_1[i] = p11[i]
            child_1.part_1[i] = p21[i]
        for i in range(p11.shape[0]):
            if p11[i] not in child_1.part_1:
                child_1.part_1[i] = p11[i]
            if p21[i] not in child_2.part_1:
                child_2.part_1[i] = p21[i]
        for i in range(p11.shape[0]):
            if child_1.part_1[i] == 0:
                child_1.part_1[i] = (rng.choice(j) for j in p11 if j not in child_1.part_1)
            if child_2.part_1[i] == 0:
                child_2.part_1[i] = (rng.choice(j) for j in p21 if j not in child_2.part_1)
        child_1.part_2 = np.sort(rng.choice(np.arange(1,max(p11)),max(p12)-1,replace=False))
        child_1.part_2 = np.sort(rng.choice(np.arange(1,max(p21)),max(p22)-1,replace=False))
    
    return (child_1,child_2)


if __name__=="__main__":
    pass

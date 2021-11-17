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
        c_1,s_1 = p1.cities,p1.tours
        c_2,s_2 = p2.cities,p2.tours
        child_1 = Chromosome_1()
        child_1.cities = np.zeros(shape=c_1.shape,dtype=int)
        child_1.tours = np.zeros(shape=s_1.shape,dtype=int)
        child_2 = Chromosome_1()
        child_2.cities = np.zeros(shape=c_2.shape,dtype=int)
        child_2.tours = np.zeros(shape=s_2.shape,dtype=int)
        cut_points = np.sort(rng.choice(c_1.shape[0],2,replace=False))
        for i in range(cut_points[0],cut_points[1]):
            child_2.cities[i] = c_1[i]
            child_2.tours[i] = s_1[i]
            child_1.cities[i] = c_2[i]
            child_1.tours[i] = s_2[i]
        for i in np.concatenate((np.arange(cut_points[0]),np.arange(cut_points[1],c_1.shape[0]))):
            if c_1[i] not in child_1.cities:
                child_1.cities[i] = c_1[i]
                child_1.tours[i] = s_1[i]
            else:
                child_1.cities[i] = rng.choice([j for j in range(1,c_1.shape[0]+1) if j not in child_1.cities])
                if np.any(np.arange(1,max(s_1)+1)) not in child_1.tours:
                    child_1.tours[i] = rng.choice([j for j in s_1 if j not in child_1.tours])
                else:
                    child_1.tours[i] = rng.choice(np.arange(1,max(s_1)+1))
            if c_2[i] not in child_2.cities:
                child_2.cities[i] = c_2[i]
                child_2.tours[i] = s_2[i]
            else:
                child_2.cities[i] = rng.choice([j for j in range(1,c_2.shape[0]+1) if j not in child_2.cities])
                if np.any(np.arange(1,max(s_2)+1)) not in child_2.tours:
                    child_2.tours[i] = rng.choice([j for j in s_2 if j not in child_2.tours])
                else:
                    child_2.tours[i] = rng.choice(np.arange(1,max(s_2)+1))
        if np.all(child_1.cities==c_1) or np.all(child_2.cities==c_2) or np.all(child_1.cities==c_2) or np.all(child_2.cities==c_1):
            child_1,child_2 = partiallyMappedCrossover(p1,p2,ctype)
                    
    if ctype==2:
        p11 = p1.part_1
        p21 = p2.part_1
        p12 = p1.part_2
        p22 = p2.part_2
        child_1 = Chromosome_2()
        child_1.part_1 = np.zeros(shape=p11.shape,dtype=int)
        child_1.part_2 = np.zeros(shape=p12.shape,dtype=int)
        child_2 = Chromosome_2()
        child_2.part_1 = np.zeros(shape=p21.shape,dtype=int)
        child_2.part_2 = np.zeros(shape=p22.shape,dtype=int)
        cut_points = np.sort(rng.choice(p11.shape[0],2,replace=False))
        for i in range(cut_points[0],cut_points[1]):
            child_2.part_1[i] = p11[i]
            child_1.part_1[i] = p21[i]
        for i in range(p11.shape[0]):
            if p11[i] not in child_1.part_1 and child_1.part_1[i]==0:
                child_1.part_1[i] = p11[i]
            if p21[i] not in child_2.part_1 and child_2.part_1[i]==0:
                child_2.part_1[i] = p21[i]
        for i in range(p11.shape[0]):
            if child_1.part_1[i] == 0:
                child_1.part_1[i] = rng.choice([j for j in p11 if j not in child_1.part_1])
            if child_2.part_1[i] == 0:
                child_2.part_1[i] = rng.choice([j for j in p21 if j not in child_2.part_1])
        child_1.part_2 = np.sort(rng.choice(np.arange(1,max(p11)),p12.shape[0],replace=False))
        child_2.part_2 = np.sort(rng.choice(np.arange(1,max(p21)),p22.shape[0],replace=False))
    
        if np.all(child_1.part_1==p21) or np.all(child_2.part_1==p21) or np.all(child_1.part_1==p11) or np.all(child_2.part_1==p11):
            child_1,child_2 = partiallyMappedCrossover(p1,p2,ctype)
    
    
    return (child_1,child_2)

def getCxChild1(a,b,c,d,child_a,child_b,start_id):
    count = 0
    i = copy.deepcopy(start_id)
    child_a = np.zeros(shape=a.shape,dtype=int)
    child_b = np.zeros(shape=b.shape,dtype=int)
    while True:
        child_a[i] = a[i]
        child_b[i] = b[i]
        k = copy.deepcopy(i)
        i = next(j for j in range(a.shape[0]) if c[k]==a[j])
        count += 1
        if count==a.shape[0]:
            return (child_a,child_b)
        if c[k]==a[start_id]:
            for j in range(a.shape[0]):
                if child_a[j]==0:
                    child_a[j] = c[j]
                    child_b[j] = d[j]
            return (child_a,child_b)

def getCxChild2(a,c,child_a,start_id):
    count = 0
    i = copy.deepcopy(start_id)
    child_a = np.zeros(shape=a.shape,dtype=int)
    while True:
        child_a[i] = a[i]
        k = copy.deepcopy(i)
        i = next(j for j in range(a.shape[0]) if c[k]==a[j])
        count += 1
        if count==a.shape[0]:
            return child_a
        if c[k]==a[start_id]:
            for j in range(a.shape[0]):
                if child_a[j]==0:
                    child_a[j] = c[j]
            return child_a
        
def cyclicCrossover(p1,p2,ctype=1):
    if ctype==1:
        c_1,s_1 = p1.cities,p1.tours
        c_2,s_2 = p2.cities,p2.tours
        child_1 = Chromosome_1()
        child_1.cities = np.empty(shape=c_1.shape,dtype=int)
        child_1.tours = np.empty(shape=s_1.shape,dtype=int)
        child_2 = Chromosome_1()
        child_2.cities = np.empty(shape=c_2.shape,dtype=int)
        child_2.tours = np.empty(shape=s_2.shape,dtype=int)
        start_id = 0
        child_1.cities,child_1.tours = getCxChild1(c_1,s_1,c_2,s_2,child_1.cities,child_1.tours,start_id)
        child_2.cities,child_2.tours = getCxChild1(c_2,s_2,c_1,s_1,child_2.cities,child_2.tours,start_id)
        
    if ctype==2:
        rng = np.random.default_rng(SEED)
        updateSeed()
        p11 = p1.part_1
        p21 = p2.part_1
        p12 = p1.part_2
        p22 = p2.part_2
        child_1 = Chromosome_2()
        child_1.part_1 = np.empty(shape=p11.shape,dtype=int)
        child_1.part_2 = np.empty(shape=p12.shape,dtype=int)
        child_2 = Chromosome_2()
        child_2.part_1 = np.empty(shape=p21.shape,dtype=int)
        child_2.part_2 = np.empty(shape=p22.shape,dtype=int)
        start_id = 0
        child_1.part_1 = getCxChild2(p11,p21,child_1.part_1,start_id)
        child_2.part_1 = getCxChild2(p21,p11,child_2.part_1,start_id)
        child_1.part_2 = np.sort(rng.choice(np.arange(1,max(p11)),p12.shape[0],replace=False))
        child_2.part_2 = np.sort(rng.choice(np.arange(1,max(p21)),p22.shape[0],replace=False))
        
    return (child_1,child_2)

def orderedCrossover(p1,p2,ctype=1):
    rng = np.random.default_rng(SEED)
    updateSeed()
    if ctype==1:
        c_1,s_1 = p1.cities,p1.tours
        c_2,s_2 = p2.cities,p2.tours
        child_1 = Chromosome_1()
        child_1.cities = np.zeros(shape=c_1.shape,dtype=int)
        child_1.tours = np.zeros(shape=s_1.shape,dtype=int)
        child_2 = Chromosome_1()
        child_2.cities = np.zeros(shape=c_2.shape,dtype=int)
        child_2.tours = np.zeros(shape=s_2.shape,dtype=int)
        cut_points = np.sort(rng.choice(c_1.shape[0],2,replace=False))
        print("cut points:", cut_points)
        for i in range(cut_points[0],cut_points[1]):
            child_1.cities[i] = c_1[i]
            child_1.tours[i] = s_1[i]
            child_2.cities[i] = c_2[i]
            child_2.tours[i] = s_2[i]
        remnant_ids = np.concatenate((np.arange(cut_points[1],c_1.shape[0]),np.arange(cut_points[0])))
        cut_ids = np.concatenate((np.arange(cut_points[1],c_1.shape[0]),np.arange(cut_points[1])))
        rearr_c1 = [c_1[i] for i in cut_ids]
        rearr_s1 = [s_1[i] for i in cut_ids]
        rearr_c2 = [c_2[i] for i in cut_ids]
        rearr_s2 = [s_2[i] for i in cut_ids]
        rem_c1 = [i for i in rearr_c1 if i not in child_2.cities]
        rem_c2 = [i for i in rearr_c2 if i not in child_1.cities]
        rem_s1 = [rearr_s1[i] for i in range(len(rearr_s1)) if rearr_c1[i] not in child_2.cities]
        rem_s2 = [rearr_s2[i] for i in range(len(rearr_s2)) if rearr_c2[i] not in child_1.cities]
        j = 0
        for i in remnant_ids:
            child_2.cities[i] = rem_c1[j]
            child_1.cities[i] = rem_c2[j]
            child_2.tours[i] = rem_s1[j]
            child_1.tours[i] = rem_s2[j]
            j += 1
        
    if ctype==2:
        p11 = p1.part_1
        p21 = p2.part_1
        p12 = p1.part_2
        p22 = p2.part_2
        child_1 = Chromosome_2()
        child_1.part_1 = np.zeros(shape=p11.shape,dtype=int)
        child_1.part_2 = np.zeros(shape=p12.shape,dtype=int)
        child_2 = Chromosome_2()
        child_2.part_1 = np.zeros(shape=p21.shape,dtype=int)
        child_2.part_2 = np.zeros(shape=p22.shape,dtype=int)
        cut_points = np.sort(rng.choice(p11.shape[0],2,replace=False))
        for i in range(cut_points[0],cut_points[1]):
            child_1.part_1[i] = p11[i]
            child_2.part_1[i] = p21[i]
        remnant_ids = np.concatenate((np.arange(cut_points[1],p11.shape[0]),np.arange(cut_points[0])))
        cut_ids = np.concatenate((np.arange(cut_points[1],p11.shape[0]),np.arange(cut_points[1])))
        rearr_p11 = [p11[i] for i in cut_ids]
        rearr_p21 = [p21[i] for i in cut_ids]
        rem_p11 = [i for i in rearr_p11 if i not in child_2.part_1]
        rem_p21 = [i for i in rearr_p21 if i not in child_1.part_1]
        j = 0
        for i in remnant_ids:
            child_2.part_1[i] = rem_p11[j]
            child_1.part_1[i] = rem_p21[j]
            j += 1
        child_1.part_2 = np.sort(rng.choice(np.arange(1,max(p11)),p12.shape[0],replace=False))
        child_2.part_2 = np.sort(rng.choice(np.arange(1,max(p21)),p22.shape[0],replace=False))
        
    return(child_1,child_2)

def insertMutation(child,ctype=1):
    rng = np.random.default_rng(SEED)
    updateSeed()
    if ctype==1:
        c = child.cities
        s = child.tours
        mutated = Chromosome_1()
        mutated.cities = copy.deepcopy(c)
        mutated.tours = copy.deepcopy(s)
        point_1 = rng.choice(np.arange(c.shape[0]-1))
        point_2 = rng.choice(np.arange(point_1+1,c.shape[0]))
        mutated.cities = np.insert(mutated.cities,point_1+1,c[point_2])
        mutated.cities = np.delete(mutated.cities,point_2+1)
        
    if ctype==2:
        p1 = child.part_1
        p2 = child.part_2
        mutated = Chromosome_2()
        mutated.part_1 = copy.deepcopy(p1)
        mutated.part_2 = np.sort(rng.choice(np.arange(1,max(p1)),p2.shape[0],replace=False))
        point_1 = rng.choice(np.arange(p1.shape[0]-1))
        point_2 = rng.choice(np.arange(point_1+1,p1.shape[0]))
        mutated.part_1 = np.insert(mutated.part_1,point_1+1,c[point_2])
        mutated.part_1 = np.delete(mutated.part_1,point_2+1)
    
    return mutated

def swapMutation(child,ctype=1):
    rng = np.random.default_rng(SEED)
    updateSeed()
    if ctype==1:
        c = child.cities
        s = child.tours
        mutated = Chromosome_1()
        mutated.cities = copy.deepcopy(c)
        mutated.tours = copy.deepcopy(s)
        diff_tour_pairs = [[i,j] for i in range(c.shape[0]) for j in range(c.shape[0]) if s[i]!=s[j]]
        points = rng.choice(diff_tour_pairs)
        mutated.cities[points[0]],mutated.cities[points[1]] = mutated.cities[points[1]],mutated.cities[points[0]]
        mutated.tours[points[0]],mutated.tours[points[1]] = mutated.tours[points[1]],mutated.tours[points[0]]
        
    if ctype==2:
        p1 = child.part_1
        p2 = child.part_2
        mutated = Chromosome_2()
        mutated.part_1 = copy.deepcopy(p1)
        mutated.part_2 = np.sort(rng.choice(np.arange(1,max(p1)),p2.shape[0],replace=False))
        points = rng.choice(np.arange(p1.shape[0]),2,replace=False)
        mutated.part_1[points[0]],mutated.part_1[points[1]] = mutated.part_1[points[1]],mutated.part_1[points[0]]
    
    return mutated

def invertMutation(child,ctype=1):
    rng = np.random.default_rng(SEED)
    updateSeed()
    if ctype==1:
        c = child.cities
        s = child.tours
        mutated = Chromosome_1()
        mutated.cities = copy.deepcopy(c)
        mutated.tours = copy.deepcopy(s)
        points = np.sort(rng.choice(np.arange(c.shape[0]),2,replace=False))
        inverse_c = [c[i] for i in range(points[1],points[0]-1,-1)]
        for i in range(points[0],points[1]+1):
            mutated.cities[i] = inverse_c[i]
        
    if ctype==2:
        p1 = child.part_1
        p2 = child.part_2
        mutated = Chromosome_2()
        mutated.part_1 = copy.deepcopy(p1)
        mutated.part_2 = np.sort(rng.choice(np.arange(1,max(p1)),p2.shape[0],replace=False))
        points = np.sort(rng.choice(np.arange(p1.shape[0]),2,replace=False))
        inverse_p1 = [p1[i] for i in range(points[1],points[0]-1,-1)]
        for i in range(points[0],points[1]+1):
            mutated.part_1[i] = inverse_p1[i]
    
    return mutated

def scrambleMutation(child,ctype=1):
    rng = np.random.default_rng(SEED)
    updateSeed()
    if ctype==1:
        c = child.cities
        s = child.tours
        mutated = Chromosome_1()
        mutated.cities = copy.deepcopy(c)
        mutated.tours = copy.deepcopy(s)
        points = np.sort(rng.choice(np.arange(c.shape[0]),2,replace=False))
        scramble_ids = rng.permutation(np.arange(points[0],points[1]+1))
        scrambled_c = [c[i] for i in scramble_ids]
        for i in range(points[0],points[1]+1):
            mutated.cities[i] = scrambled_c[i]
        
    if ctype==2:
        p1 = child.part_1
        p2 = child.part_2
        mutated = Chromosome_2()
        mutated.part_1 = copy.deepcopy(p1)
        mutated.part_2 = np.sort(rng.choice(np.arange(1,max(p1)),p2.shape[0],replace=False))
        points = np.sort(rng.choice(np.arange(p1.shape[0]),2,replace=False))
        scramble_ids = rng.permutation(np.arange(points[0],points[1]+1))
        scrambled_p1 = [p1[i] for i in scramble_ids]
        for i in range(points[0],points[1]+1):
            mutated.part_1[i] = scrambled_p1[i]
    
    return mutated





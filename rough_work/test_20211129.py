"""
Created on Fri Nov 29 13:20:11 2021

@author: rahul
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import functools

SEED = 123456789

def updateSeed():
    global SEED
    large = 2147483647;
    k = int(SEED/127773);
    SEED = 16807*(SEED-k*127773)-k*2836;
    if SEED<=0:
        SEED += large;

def objectiveFunction1(C,X):
    total_distance = 0
    for i,tour in enumerate(X):
        distance = np.sum(np.multiply(C,tour))
        total_distance += distance
    return (-total_distance)

def objectiveFunction2(C,X,ftype='r'):
    n_tours = X.shape[0]
    tour_lengths = np.empty(shape=(n_tours,),dtype=float)
    total_distance = 0
    avg_tour_length = 0
    variation = 0
    
    if ftype=='td': #Total deviation from average
        for i,tour in enumerate(X):
            distance = np.sum(np.multiply(C,tour))
            total_distance += distance
            tour_lengths[i] = distance
        avg_tour_length = total_distance/n_tours
        for i in range(n_tours):
            variation += np.abs(avg_tour_length - tour_lengths[i])
        return (-variation)
    
    if ftype=='r': #Range of tour lengths
        for i,tour in enumerate(X):
            tour_lengths[i] = np.sum(np.multiply(C,tour))
        return -(max(tour_lengths)-min(tour_lengths))
    
    if ftype=='std': #Standard deviation
        for i,tour in enumerate(X):
            distance = np.sum(np.multiply(C,tour))
            total_distance += distance
            tour_lengths[i] = distance
        avg_tour_length = total_distance/n_tours
        for i in range(n_tours):
            variation += (avg_tour_length - tour_lengths[i])**2
        return -((variation/(n_tours-1))**0.5)
      
def getTourMatrix(chromosome,ctype=1):
    if ctype==1:
        n_tours = max(chromosome.tours) #same as no. of salespersons
        n_cities = max(chromosome.cities)
        tour_matrix = np.zeros(shape=(n_tours,n_cities+1,n_cities+1),dtype=int)
        tour_order_lists = [[] for i in range(n_tours)]
        for i in range(n_cities):
            tour_order_lists[chromosome.tours[i]-1].append(chromosome.cities[i])
        
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
        self.tours = None
        self.cities = None
        self.domination_count = None
        self.nondomination = None
        self.dominated_solutions = None
        self.crowding_distance = None
        self.function_vals = None
    def __eq__(self,other):
        if isinstance(self,other.__class__):
            return ((np.all(self.tours==other.tours)) and (np.all(self.cities==other.cities)))
        return False

class Chromosome_2(object):
    def __init__(self):
        self.part_1 = None
        self.part_2 = None
        self.domination_count = None
        self.nondomination = None
        self.dominated_solutions = None
        self.crowding_distance = None
        self.function_vals = None
    def __eq__(self,other):
        if isinstance(self,other.__class__):
            return ((np.all(self.part_1==other.part_1)) and (np.all(self.part_2==other.part_2)))
        return False
      
def createChromosome(C,n_tours,ctype=1):
    rng = np.random.default_rng(SEED)
    updateSeed()
    n_cities = C.shape[0]
    if ctype==1:
        while True:
            check = True
            chromosome = Chromosome_1()
            
            chromosome.cities = rng.permutation(np.arange(1,n_cities))
            chromosome.tours = rng.choice(np.arange(1,n_tours+1),n_cities-1)
            for i in range(1,n_tours+1):
                if i not in chromosome.tours:
                    check = False
            if check == True:
                X = getTourMatrix(chromosome,ctype)
                a = objectiveFunction1(C,X)
                b = objectiveFunction2(C,X)
                chromosome.function_vals = [a,b]
                return chromosome
    if ctype==2:
        chromosome = Chromosome_2()
        chromosome.part_1 = rng.permutation(np.arange(1,n_cities))
        chromosome.part_2 = np.sort(rng.choice(np.arange(1,n_cities-1),n_tours-1,replace=False))
        X = getTourMatrix(chromosome,ctype)
        a = objectiveFunction1(C,X)
        b = objectiveFunction2(C,X)
        chromosome.function_vals = [a,b]
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
  
def createInitialPopulation(N,C,n_tours,ctype=1):
    population = []
    while len(population)<N:
        individual = createChromosome(C,n_tours,ctype)
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
            child_2.cities[i],child_2.tours[i] = c_1[i],s_1[i]
            child_1.cities[i],child_1.tours[i] = c_2[i],s_2[i]
        for i in np.concatenate((np.arange(cut_points[0]),np.arange(cut_points[1],c_1.shape[0]))):
            if c_1[i] not in child_1.cities:
                child_1.cities[i],child_1.tours[i] = c_1[i],s_1[i]
            else:
                child_1.cities[i] = rng.choice([j for j in range(1,c_1.shape[0]+1) if j not in child_1.cities])
                if np.any(np.arange(1,max(s_1)+1)) not in child_1.tours:
                    child_1.tours[i] = rng.choice([j for j in s_1 if j not in child_1.tours])
                else:
                    child_1.tours[i] = rng.choice(np.arange(1,max(s_1)+1))
            if c_2[i] not in child_2.cities:
                child_2.cities[i],child_2.tours[i] = c_2[i],s_2[i]
            else:
                child_2.cities[i] = rng.choice([j for j in range(1,c_2.shape[0]+1) if j not in child_2.cities])
                if np.any(np.arange(1,max(s_2)+1)) not in child_2.tours:
                    child_2.tours[i] = rng.choice([j for j in s_2 if j not in child_2.tours])
                else:
                    child_2.tours[i] = rng.choice(np.arange(1,max(s_2)+1))
                    
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
            child_2.part_1[i],child_1.part_1[i] = p11[i],p21[i]
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
    
    
    return (child_1,child_2)
  
def getCxChild1(a,b,c,d,child_a,child_b,start_id):
    count = 0
    i = copy.deepcopy(start_id)
    child_a = np.zeros(shape=a.shape,dtype=int)
    child_b = np.zeros(shape=b.shape,dtype=int)
    while True:
        child_a[i],child_b[i] = a[i],b[i]
        k = copy.deepcopy(i)
        i = next(j for j in range(a.shape[0]) if c[k]==a[j])
        count += 1
        if count==a.shape[0]:
            return (child_a,child_b)
        if c[k]==a[start_id]:
            for j in range(a.shape[0]):
                if child_a[j]==0:
                    child_a[j],child_b[j] = c[j],d[j]
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
        for i in range(cut_points[0],cut_points[1]):
            child_1.cities[i],child_1.tours[i] = c_1[i],s_1[i]
            child_2.cities[i],child_2.tours[i] = c_2[i],s_2[i]
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
            child_2.cities[i],child_1.cities[i] = rem_c1[j],rem_c2[j]
            child_2.tours[i],child_1.tours[i] = rem_s1[j],rem_s2[j]
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
            child_1.part_1[i],child_2.part_1[i] = p11[i],p21[i]
        remnant_ids = np.concatenate((np.arange(cut_points[1],p11.shape[0]),np.arange(cut_points[0])))
        cut_ids = np.concatenate((np.arange(cut_points[1],p11.shape[0]),np.arange(cut_points[1])))
        rearr_p11 = [p11[i] for i in cut_ids]
        rearr_p21 = [p21[i] for i in cut_ids]
        rem_p11 = [i for i in rearr_p11 if i not in child_2.part_1]
        rem_p21 = [i for i in rearr_p21 if i not in child_1.part_1]
        j = 0
        for i in remnant_ids:
            child_2.part_1[i],child_1.part_1[i] = rem_p11[j],rem_p21[j]
            j += 1
        child_1.part_2 = np.sort(rng.choice(np.arange(1,max(p11)),p12.shape[0],replace=False))
        child_2.part_2 = np.sort(rng.choice(np.arange(1,max(p21)),p22.shape[0],replace=False))
        
    return(child_1,child_2)

def decodeChromosome2(chromosome):
    decoded = [0]
    encoded_part1 = copy.deepcopy(chromosome.part_1)
    encoded_part2 = copy.deepcopy(chromosome.part_2)
    count = 0
    for i,val in enumerate(encoded_part1):
        if i!=encoded_part2[count]:
            decoded.append(val)
        else:
            decoded.append(0)
            decoded.append(val)
            if count!=len(encoded_part2)-1:
                count+=1
    return decoded

def rationalizeHgaResult(org_result):
    if org_result[0]!=0:
        i = org_result.index(0)
        cut_out = org_result[:i]
        cut_out.reverse()
        new_result = [j for j in org_result[i:] if j != 0]
        new_result.extend(cut_out)
    else:
        new_result = [j for j in org_result if j!=0]
    return new_result

def heirarchicalCrossover(p1,p2,C):
    rng = np.random.default_rng(SEED)
    updateSeed()
    child_1 = Chromosome_2()
    child_2 = Chromosome_2()
    p11 = p1.part_1.tolist()
    p21 = p2.part_1.tolist()
    dp1 = decodeChromosome2(p1)
    dp2 = decodeChromosome2(p2)
    k = rng.choice(p11)
    result_1 = [k]
    while len(p11)>1:
        i = p11.index(k)
        j = p21.index(k)
        cities = []
        left_city_1,right_city_1 = p11[i-1],p11[(i+1)%len(p11)]
        left_city_2,right_city_2 = p21[j-1],p21[(j+1)%len(p21)]
        
        if i==len(p11)-1:
            cities.append(left_city_1)
        else:
            cities.append(right_city_1)
        if j==len(p21)-1:
            cities.append(left_city_2)
        else:
            cities.append(right_city_2)
        distances = [C[k,cities[0]],C[k,cities[1]]]
        p11.remove(k)
        p21.remove(k)
        k = cities[np.argsort(distances)[0]]
        result_1.append(k)
    k = rng.choice(p1.part_1)
    result_2 = [k]
    while len(dp1)>1:
        i = dp1.index(k)
        j = dp2.index(k)
        cities =[]
        left_city_1,right_city_1 = dp1[i-1],dp1[(i+1)%len(dp1)]
        left_city_2,right_city_2 = dp2[j-1],dp2[(j+1)%len(dp2)]
        
        if i==len(dp1)-1:
            cities.append(left_city_1)
        else:
            cities.append(right_city_1)
        if j==len(dp2)-1:
            cities.append(left_city_2)
        else:
            cities.append(right_city_2)
        dp1.remove(k)
        dp2.remove(k)
        if C[k,cities[0]]>C[k,cities[1]]:
            k = cities[1]
        else:
            k = cities[0]
        result_2.append(k)
    result_2 = rationalizeHgaResult(result_2)
    child_1.part_1 = np.array(result_1)
    child_2.part_1 = np.array(result_2)
    copy_choice = rng.choice([0,1])
    if copy_choice==0:
        child_1.part_2 = copy.deepcopy(p1.part_2)
    else:
        child_1.part_2 = copy.deepcopy(p2.part_2)
    child_2.part_2 = np.sort(rng.choice(np.arange(1,max(p2.part_1)),p2.part_2.shape[0],replace=False))
    return child_1,child_2

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
        mutated.part_1 = np.insert(mutated.part_1,point_1+1,p1[point_2])
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
        if diff_tour_pairs == []:
            mutated = child
            return mutated
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
        j = 0
        for i in range(points[1],points[0]-1,-1):
            mutated.cities[i] = inverse_c[j]
            j += 1
        
    if ctype==2:
        p1 = child.part_1
        p2 = child.part_2
        mutated = Chromosome_2()
        mutated.part_1 = copy.deepcopy(p1)
        mutated.part_2 = np.sort(rng.choice(np.arange(1,max(p1)),p2.shape[0],replace=False))
        points = np.sort(rng.choice(np.arange(p1.shape[0]),2,replace=False))
        inverse_p1 = [p1[i] for i in range(points[1],points[0]-1,-1)]
        j = 0
        for i in range(points[1],points[0]-1,-1):
            mutated.part_1[i] = inverse_p1[j]
            j += 1
    
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
        for i,scrambled_id in enumerate(scramble_ids):
            mutated.cities[scrambled_id] = scrambled_c[i]
        
    if ctype==2:
        p1 = child.part_1
        p2 = child.part_2
        mutated = Chromosome_2()
        mutated.part_1 = copy.deepcopy(p1)
        mutated.part_2 = np.sort(rng.choice(np.arange(1,max(p1)),p2.shape[0],replace=False))
        points = np.sort(rng.choice(np.arange(p1.shape[0]),2,replace=False))
        scramble_ids = rng.permutation(np.arange(points[0],points[1]+1))
        scrambled_p1 = [p1[i] for i in scramble_ids]
        for i,scrambled_id in enumerate(scramble_ids):
            mutated.part_1[scrambled_id] = scrambled_p1[i]
    
    return mutated
  
def dominates(individual,other,ctype=1):
    value = False
    if (individual.function_vals[0]>other.function_vals[0]) or (individual.function_vals[1]>other.function_vals[1]):
        if (individual.function_vals[0]>=other.function_vals[0]) and (individual.function_vals[1]>=other.function_vals[1]):
            value = True
    return value
  
def nondominatedSort(population,ctype):
    nondominated_fronts = [[]]
    for individual in population:
        individual.dominated_solutions = []
        individual.domination_count = 0
        for other in population:
            if dominates(individual,other,ctype):
                individual.dominated_solutions.append(other)
            elif dominates(other,individual,ctype):
                individual.domination_count += 1
        if individual.domination_count == 0:
            individual.nondomination = 1
            nondominated_fronts[0].append(individual)
    i = 0
    while len(nondominated_fronts[i])>0:
        Q = []
        for individual in nondominated_fronts[i]:
            for dominated in individual.dominated_solutions:
                dominated.domination_count -= 1
                if dominated.domination_count == 0:
                    dominated.nondomination = i+2
                    Q.append(dominated)
        i += 1
        nondominated_fronts.append(Q)
    return nondominated_fronts
  
def crowdedComparisonOperator(i,j):
    if i.nondomination<j.nondomination or (i.nondomination==j.nondomination and i.crowding_distance>j.crowding_distance):
        return -1
    elif j.nondomination<i.nondomination or (i.nondomination==j.nondomination and j.crowding_distance>i.crowding_distance):
        return 1
    else:
        return 0
      
def tournamentSelection(population,tsub,selection_probability,ctype=1):
    rng = np.random.default_rng(SEED)
    updateSeed()
    parents = []
    probability_list = [selection_probability]
    best = None
    for i in range(1,tsub):
        probability_list.append(selection_probability*(1-selection_probability)**i)
    while len(parents)<len(population):
        tournament_bracket = rng.choice(population,tsub,replace=False).tolist()
        tournament_bracket = sorted(tournament_bracket,key=functools.cmp_to_key(crowdedComparisonOperator))
        for i in range(tsub):
            best = tournament_bracket[tsub-1]
            if rng.choice([0,1],p=[probability_list[i],1-probability_list[i]])==0:
                if parents==[] or parents[-1]!=tournament_bracket[i]:
                    best = tournament_bracket[i]
                    break
        parents.append(best)
    return parents

def fDistance(a,b):
    fval_a = a.function_vals
    fval_b = b.function_vals
    return euclideanDistance(fval_a, fval_b)

def tournamentSelection2(pop,tsub=2,ctype=1):
    population = copy.deepcopy(pop)
    rng = np.random.default_rng(SEED)
    updateSeed()
    parents = []
    farthest = None
    while len(parents)<len(population):
        tournament_bracket = rng.choice(population,tsub,replace=False).tolist()
        tournament_bracket = sorted(tournament_bracket,key=functools.cmp_to_key(crowdedComparisonOperator))
        parents.append(tournament_bracket[0])
        d_list = [fDistance(ind,tournament_bracket[0]) for ind in population]
        sorted_ids = np.argsort(d_list)
        farthest = population[sorted_ids[-1]]
        parents.append(farthest)
        population.remove(tournament_bracket[0])
        population.remove(farthest)
    return parents

def createOffspringPopulation(population,dist_matrix,tsub=2,selection_prob=0.9,cxtype='pmx',mu_prob=0.05,ctype=1):
    rng = np.random.default_rng(SEED)
    updateSeed()
    children = []
    child1 = None
    child2 = None
    parents = tournamentSelection(population,tsub,selection_prob,ctype)
    #parents = tournamentSelection2(population,tsub,ctype)
    mating_pairs = [(parents[i],parents[i+1]) for i in range(len(parents)) if i%2==0]
    for pair in mating_pairs:
        if cxtype=='pmx':
            child1,child2 = partiallyMappedCrossover(pair[0],pair[1],ctype)
        elif cxtype=='cycx':
            child1,child2 = cyclicCrossover(pair[0],pair[1],ctype)
        elif cxtype=='hx':
            child1,child2 = heirarchicalCrossover(pair[0],pair[1],dist_matrix)
        else:
            child1,child2 = orderedCrossover(pair[0],pair[1],ctype)
        
        if rng.choice([0,1],p=[mu_prob,1-mu_prob])==0:
            child1 = mutate_child(child1,ctype)
        #if rng.choice([0,1],p=[mu_prob,1-mu_prob])==0:
            child2 = mutate_child(child2,ctype)
        X = getTourMatrix(child1,ctype)
        a = objectiveFunction1(dist_matrix,X)
        b = objectiveFunction2(dist_matrix,X)
        child1.function_vals = [a,b]
        Y = getTourMatrix(child2,ctype)
        a = objectiveFunction1(dist_matrix,Y)
        b = objectiveFunction2(dist_matrix,Y)
        child2.function_vals = [a,b]
        
        children.append(child1)
        children.append(child2)
    return children
  
def mutate_child(child,ctype):
    rng = np.random.default_rng(SEED)
    updateSeed()
    mutated = None
    mu_type = rng.choice([0,1,2,3])
    if mu_type==0:
        mutated = insertMutation(child,ctype)
    elif mu_type==1:
        mutated = swapMutation(child,ctype)
    elif mu_type==2:
        mutated = invertMutation(child,ctype)
    else:
        mutated = scrambleMutation(child,ctype)
    return mutated
  
def assignCrowdingDistance(list_of_individuals):
    list_I = copy.deepcopy(list_of_individuals)
    l = len(list_I)
    for i in list_I:
        i.crowding_distance = 0
    n_objs = len(list_I[0].function_vals)
    for i in range(n_objs):
        list_I.sort(key=lambda ind:ind.function_vals[i])
        fmax = list_I[-1].function_vals[i]
        fmin = list_I[0].function_vals[i]
        list_I[0].crowding_distance = 10**9
        list_I[-1].crowding_distance = 10**9
        for j in range(1,l-1):
            list_I[j].crowding_distance += (list_I[j+1].function_vals[i]-
                                                      list_I[j-1].function_vals[i])/(fmax-fmin)
    return list_I

def readInstance(filename):
    city_coordinates = []
    with open(filename) as f:
        for line in f.readlines():
            int_list = [int(float(i)) for i in line.split()]
            city_coordinates.append((int_list[1],int_list[2]))
    distance_matrix = np.empty(shape=(len(city_coordinates),len(city_coordinates)),dtype=int)
    for i,city_i in enumerate(city_coordinates):
        for j,city_j in enumerate(city_coordinates):
            distance_matrix[i,j] = round(euclideanDistance(city_i,city_j))
    #Plot generate instance:
    plt.figure()
    x = [p[0] for p in city_coordinates]
    y = [p[1] for p in city_coordinates]
    plt.plot(x,y,"ob")
    plt.plot(city_coordinates[0][0],city_coordinates[0][1],"or")
    plt.grid()
    plt.show()
    return distance_matrix

def main():
    n_iters = 250
    n_cities = 51
    n_tours = 7
    #map_size = 500
    ctype = 2
    pop_size = 100
    tsub = 2
    selection_probability = 1
    cx_type = 'hx'
    mutation_probability = 0.05
    print("***Multi-Objective Problem: MOmTSP with",n_cities,"cities and",n_tours,"salespersons***")
    print("Algorithm: NSGA_II. Crossover type>",cx_type,";Mutation Probability>",mutation_probability,"n(iterations)>",n_iters,".")
    print("\n-------- PROGRAM START ---------\n")
    #print("Generating Instance...")
    #C = generateInstance(n_cities,map_size)
    print("Reading instance from file...")
    C = readInstance('eil51.txt')
    print("...Done\n")
    print("Creating initial population...")
    population = createInitialPopulation(pop_size,C,n_tours,ctype)
    print("...Done\n")
    extra_front = []
    first_front = []
    print(">>>Entering Main Loop:\n")
    for iter_count in range(n_iters):
        print("------Iteration no.->",iter_count+1,"------\n")
        print("   Doing non-dominated sort...")
        fronts = nondominatedSort(population,ctype)
        print("   ...Done.")
        print("   Getting P(t+1)...")
        next_generation_P = []
        i = 0
        while True:
            if len(next_generation_P)+len(fronts[i])>=pop_size:
                break
            crowding_assigned_front = assignCrowdingDistance(fronts[i])
            next_generation_P.extend(crowding_assigned_front)
            i += 1
        if len(next_generation_P)<pop_size:
            P_temp_length = len(next_generation_P)
            extra_front = assignCrowdingDistance(fronts[i])
            if len(extra_front)>1:
                extra_front = sorted(extra_front,key=functools.cmp_to_key(crowdedComparisonOperator))
            next_generation_P.extend(extra_front[0:pop_size-P_temp_length])
        print("   ...Done.")
        print("   Creating Offspring Q(t+1)...")
        next_generation_Q = createOffspringPopulation(next_generation_P,C,tsub,selection_probability,
                                                      cx_type,mutation_probability,ctype)
        print("   ...Done.")
        print("   Getting R(t+1) = P(t+1) U Q(t+1)...")
        population = next_generation_P
        population.extend(next_generation_Q)
        print("   ...Done.\n")
        #plt.clf()
        first_front = fronts[0]
        second_front = fronts[1]
        third_front = fronts[2]
        #fvalues = [(i.function_vals[0],i.function_vals[1]) for i in first_front]
        #X = [-i[0] for i in fvalues]
        #Y = [-i[1] for i in fvalues]
        #plt.figure()
        #plt.xlabel("Total Distance")
        #plt.ylabel("Average Tour Distance")
        #plt.scatter(X,Y)
        #plt.show()
    print(">>>Exited Main Loop")
    return first_front,second_front,third_front
        
if __name__=="__main__":
    best_front,niban,sanban = main()
    print("Best solution front after all iterations in figure\n")
    fvalues1 = [(i.function_vals[0],i.function_vals[1]) for i in best_front]
    fvalues2 = [(i.function_vals[0],i.function_vals[1]) for i in niban]
    fvalues3 = [(i.function_vals[0],i.function_vals[1]) for i in sanban]
    X = [-i[0] for i in fvalues1]
    Y = [-i[1] for i in fvalues1]
    plt.figure()
    plt.xlabel("Total Distance")
    plt.ylabel("Average Tour Distance")
    plt.scatter(X,Y,c='b')
    X = [-i[0] for i in fvalues2]
    Y = [-i[1] for i in fvalues2]
    plt.scatter(X,Y,c='r')
    X = [-i[0] for i in fvalues3]
    Y = [-i[1] for i in fvalues3]
    plt.scatter(X,Y,c='k')
    plt.show()

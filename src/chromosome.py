"""
chromosome.py (module)

Written as part of the submission for ECE 750 AL: Bio&Comp Individual Project
Author: Rahul Balamurugan

Contains the chromosome class definition(s), and functions to create chromosome
as well as initial population. Does nothing if executed.

List of contained Classes:
    
    Chromosome_1():
        2 string chromosome representation.
    
    Chromosome_2():
        1 string with breakpoints chromosome representation.

List of contained functions:
    
    createChromosome():
        
        
    createInitialPopulation(N,C,T,coords,n_tours,ctype=1):
        Returns list of new individuals (objects), each generated by calling
        the createChromosome() function from module 'chromosome'

"""

import numpy as np
import src.utils as utils
import src.mtsp as mtsp

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
            return ((np.all(self.tours==other.tours)) and 
                    (np.all(self.cities==other.cities)))
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
            return ((np.all(self.part_1==other.part_1)) and 
                    (np.all(self.part_2==other.part_2)))
        return False

def createChromosome(C,T,n_tours,ctype=1,ptype=1):
    rng = np.random.default_rng()
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
                X = utils.getTourMatrix(chromosome,ctype)
                a = mtsp.objectiveFunction1(C,X)
                b = mtsp.objectiveFunction2(C,T,X,ptype)
                chromosome.function_vals = [a,b]
                return chromosome
    if ctype==2:
        chromosome = Chromosome_2()
        chromosome.part_1 = rng.permutation(np.arange(1,n_cities))
        chromosome.part_2 = np.sort(rng.choice(np.arange(1,n_cities-1),
                                               n_tours-1,replace=False))
        X = utils.getTourMatrix(chromosome,ctype)
        a = mtsp.objectiveFunction1(C,X)
        b = mtsp.objectiveFunction2(C,T,X,ptype)
        chromosome.function_vals = [a,b]
        return chromosome
    
def createInitialPopulation(N,C,T,data,n_tours,ctype=1,ptype=1):
    population = []
    while len(population)<N:
        individual = createChromosome(C,T,n_tours,ctype,ptype)
        if individual not in population:
            population.append(individual)
    return population

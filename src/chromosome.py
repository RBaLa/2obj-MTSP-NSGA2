import numpy as np
import utils
import mtsp

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
      
def createChromosome(C,n_tours,ctype=1):
    rng = np.random.default_rng(seed.SEED)
    utils.updateSeed()
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
                b = mtsp.objectiveFunction2(C,X)
                chromosome.function_vals = [a,b]
                return chromosome
    if ctype==2:
        chromosome = Chromosome_2()
        chromosome.part_1 = rng.permutation(np.arange(1,n_cities))
        chromosome.part_2 = np.sort(rng.choice(np.arange(1,n_cities-1),
                                               n_tours-1,replace=False))
        X = utils.getTourMatrix(chromosome,ctype)
        a = objectiveFunction1(C,X)
        b = objectiveFunction2(C,X)
        chromosome.function_vals = [a,b]
        return chromosome

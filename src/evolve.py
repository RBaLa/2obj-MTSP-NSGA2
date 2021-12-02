import numpy as np
import chromosome as chrom
import utils
import functools
import genops
import nsga2
import seed

def createInitialPopulation(N,C,data,n_tours,ctype=1):
    population = []
    while len(population)<N:
        individual = chrom.createChromosome(C,n_tours,ctype)
        #individual = chrom.createChromosomeFromTspSolution(C,data,n_tours)
        if individual not in population:
            population.append(individual)
        
    return population
  
  def tournamentSelection(population,tsub,selection_probability,ctype=1):
    rng = np.random.default_rng(seed.SEED)
    utils.updateSeed()
    parents = []
    probability_list = [selection_probability]
    best = None
    for i in range(1,tsub):
        probability_list.append(selection_probability*
                                (1-selection_probability)**i)
    while len(parents)<len(population):
        tournament_bracket = rng.choice(population,tsub,replace=False).tolist()
        tournament_bracket = sorted(tournament_bracket,
                                    key=functools.cmp_to_key(
                                        nsga2.crowdedComparisonOperator))
        for i in range(tsub):
            best = tournament_bracket[tsub-1]
            if rng.choice([0,1],
                          p=[probability_list[i],1-probability_list[i]])==0:
                if parents==[] or parents[-1]!=tournament_bracket[i]:
                    best = tournament_bracket[i]
                    break
        parents.append(best)
    return parents

def fDistance(a,b):
    fval_a = a.function_vals
    fval_b = b.function_vals
    return utils.euclideanDistance(fval_a, fval_b)

def tournamentSelection2(pop,tsub=2,ctype=1):
    population = copy.deepcopy(pop)
    rng = np.random.default_rng(seed.SEED)
    utils.updateSeed()
    parents = []
    farthest = None
    while len(parents)<len(population):
        tournament_bracket = rng.choice(population,tsub,replace=False).tolist()
        tournament_bracket = sorted(tournament_bracket,
                                    key=functools.cmp_to_key(
                                        nsga2.crowdedComparisonOperator))
        parents.append(tournament_bracket[0])
        d_list = [fDistance(ind,tournament_bracket[0]) for ind in population]
        sorted_ids = np.argsort(d_list)
        farthest = population[sorted_ids[-1]]
        parents.append(farthest)
        population.remove(tournament_bracket[0])
        population.remove(farthest)
    return parents

def createOffspringPopulation(population,dist_matrix,tsub=2,selection_prob=0.9,
                              cxtype='pmx',mu_prob=0.05,ctype=1):
    rng = np.random.default_rng(seed.SEED)
    utils.updateSeed()
    children = []
    child1 = None
    child2 = None
    parents = tournamentSelection(population,tsub,selection_prob,ctype)
    #parents = tournamentSelection2(population,tsub,ctype)
    mating_pairs = [(parents[i],parents[i+1]) 
                    for i in range(len(parents)) if i%2==0]
    for pair in mating_pairs:
        if cxtype=='pmx':
            child1,child2 = genops.partiallyMappedCrossover(pair[0],pair[1],ctype)
        elif cxtype=='cycx':
            child1,child2 = genops.cyclicCrossover(pair[0],pair[1],ctype)
        elif cxtype=='hx':
            child1,child2 = genops.heirarchicalCrossover(pair[0],pair[1],dist_matrix)
        else:
            child1,child2 = genops.orderedCrossover(pair[0],pair[1],ctype)
        
        if rng.choice([0,1],p=[mu_prob,1-mu_prob])==0:
            child1 = genops.mutate_child(child1,ctype)
        if rng.choice([0,1],p=[mu_prob,1-mu_prob])==0:
            child2 = genops.mutate_child(child2,ctype)
        X = utils.getTourMatrix(child1,ctype)
        a = mtsp.objectiveFunction1(dist_matrix,X)
        b = mtsp.objectiveFunction2(dist_matrix,X)
        child1.function_vals = [a,b]
        Y = utils.getTourMatrix(child2,ctype)
        a = mtsp.objectiveFunction1(dist_matrix,Y)
        b = mtsp.objectiveFunction2(dist_matrix,Y)
        child2.function_vals = [a,b]
        
        children.append(child1)
        children.append(child2)
    return children

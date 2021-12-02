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
        if individual not in population:
            population.append(individual)
        
    return population
  
  def tournamentSelection(population,selection_probability,ctype=1):
    rng = np.random.default_rng(seed.SEED)
    utils.updateSeed()
    parents = []
    probability_list = [selection_probability]
    best = None
    tsub = 2
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

def createOffspringPopulation(population,dist_matrix,selection_prob=0.9,
                              cxtype='pmx',mu_prob=0.05,ctype=1):
    rng = np.random.default_rng(seed.SEED)
    utils.updateSeed()
    children = []
    child1 = None
    child2 = None
    parents = tournamentSelection(population,selection_prob,ctype)
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

def evolve(n_iters,population,C,selection_probability,cx_type,mutation_probability,ctype):
    extra_front = []
    fronts = []
    print("   >>>Entering Main Loop:\n")
    for iter_count in tqdm.tqdm(range(n_iters)):
        fronts = nsga2.nondominatedSort(population,ctype)
        next_generation_P = []
        i = 0
        while True:
            if len(next_generation_P)+len(fronts[i])>=pop_size:
                break
            crowding_assigned_front = nsga2.assignCrowdingDistance(fronts[i])
            next_generation_P.extend(crowding_assigned_front)
            i += 1
        if len(next_generation_P)<pop_size:
            P_temp_length = len(next_generation_P)
            extra_front = nsga2.assignCrowdingDistance(fronts[i])
            if len(extra_front)>1:
                extra_front = sorted(extra_front,
                                     key=functools.cmp_to_key(
                                         nsga2.crowdedComparisonOperator))
            next_generation_P.extend(extra_front[0:pop_size-P_temp_length])
        next_generation_Q = createOffspringPopulation(next_generation_P,C,
                                                      selection_probability,
                                                      cx_type,
                                                      mutation_probability,
                                                      ctype)
        population = next_generation_P
        population.extend(next_generation_Q)
    return fronts

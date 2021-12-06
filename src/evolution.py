# -*- coding: utf-8 -*-
"""
evolution.py (module)

Written as part of the submission for ECE 750 AL: Bio&Comp Individual Project
Author: Rahul Balamurugan

Contains the functions that do the offspring generation. Does nothing if 
executed.

List of functions:
    createOffspringPopulation(population,dist_matrix,T,tsub=2,selection_prob=1,
                              cxtype='hx',mu_prob=0.05,ctype=1):
        Returns offspring popullation from input parent population. Calls
        functions from most other modules. Is called only by evolve().
        
    evolve(pop_size,C,T,data,n_tours,ctype,cx_type,tsub,selection_probability,
           mutation_probability,n_iters,n_reps):
        Returns optimal pareto front after running the NSGA-II algorithm with
        argument parameters for n_reps times and combining the obtained
        optimal pareto fronts from each run.
        
"""

import numpy as np
import tqdm
import functools
import src.genops as op
import src.mtsp as mtsp
import src.utils as u
import src.chromosome as chrom
import src.nsga2 as nsga2

def createOffspringPopulation(population,dist_matrix,T,tsub=2,selection_prob=1,
                              cxtype='hx',mu_prob=0.05,ctype=1,ptype=1):
    rng = np.random.default_rng()
    children = []
    child1 = None
    child2 = None
    parents = op.tournamentSelection(population,tsub,selection_prob)
    mating_pairs = [(parents[i],parents[i+1]) 
                    for i in range(len(parents)) if i%2==0]
    for pair in mating_pairs:
        if cxtype=='pmx':
            child1,child2 = op.partiallyMappedCrossover(pair[0],pair[1])
        elif cxtype=='cycx':
            child1,child2 = op.cyclicCrossover(pair[0],pair[1],ctype)
        elif cxtype=='hx':
            child1,child2 = op.hierarchicalCrossover(pair[0],pair[1],dist_matrix,T,ptype=1)
        else:
            child1,child2 = op.orderedCrossover(pair[0],pair[1],ctype)
        
        if rng.choice([0,1],p=[mu_prob,1-mu_prob])==0:
            child1 = op.mutateChild(child1,ctype)
        if rng.choice([0,1],p=[mu_prob,1-mu_prob])==0:
            child2 = op.mutateChild(child2,ctype)
        X = u.getTourMatrix(child1,ctype)
        a = mtsp.objectiveFunction1(dist_matrix,X)
        b = mtsp.objectiveFunction2(dist_matrix,T,X,ptype)
        child1.function_vals = [a,b]
        Y = u.getTourMatrix(child2,ctype)
        a = mtsp.objectiveFunction1(dist_matrix,Y)
        b = mtsp.objectiveFunction2(dist_matrix,T,Y,ptype)
        child2.function_vals = [a,b]
        
        children.append(child1)
        children.append(child2)
    return children

def evolve(pop_size,C,T,data,n_tours,n_iters,cx_type='hx',ctype=2,
           selection_probability=1,mutation_probability=0.05,tsub=2,n_reps=10,
           ptype=1):
    population = chrom.createInitialPopulation(pop_size,C,T,data,n_tours,ctype,ptype)
    extra_front = []
    first_front = []
    best_fronts = []
    fronts = []
    print(">>>Entering Main Loop:\n")
    for rep_count in range(n_reps):
        print("RUN NUMBER",rep_count+1)
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
                                                          T,tsub,
                                                          selection_probability,
                                                          cx_type,
                                                          mutation_probability,
                                                          ctype,ptype)
            population = next_generation_P
            population.extend(next_generation_Q)
            first_front = fronts[0]
        best_fronts.extend(first_front)
    return best_fronts,fronts
def dominates(individual,other,ctype=1):
    value = False
    if (individual.function_vals[0]>other.function_vals[0]) or (
            individual.function_vals[1]>other.function_vals[1]):
        if (individual.function_vals[0]>=other.function_vals[0]) and (
                individual.function_vals[1]>=other.function_vals[1]):
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
    if i.nondomination<j.nondomination or (i.nondomination==j.nondomination 
                                and i.crowding_distance>j.crowding_distance):
        return -1
    elif j.nondomination<i.nondomination or (i.nondomination==j.nondomination 
                                and j.crowding_distance>i.crowding_distance):
        return 1
    else:
        return 0
      
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
                                            list_I[j-1].function_vals[i])/(
                                                fmax-fmin)
    return list_I

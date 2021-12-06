"""
genops.py (module)

Written as part of the submission for ECE 750 AL: Bio&Comp Individual Project
Author: Rahul Balamurugan

Contains the selection, crossover, and mutation genetic operators as well as 
any specific functions solely called by the operators. Does nothing if
executed.

List of functions:
    
    tournamentSelection(population,tsub=2,selection_probability=1):
        Returns ordered list of winning parents where every consecutive pair
        is a mating pair. Does tournament selection using the crowding
        distance and non-domination level as comparators. Default number of
        competing individuals is 2 (Binary tournament), and default selection
        probability is 1.
    
    partiallyMappedCrossover(p1,p2,ctype):
        Returns tuple of (child1,child2) by crossing p1 and p2 using the 
        Partially Mapped Crossover method. Can do for either chromosome
        representation.
        
    getCxChild1():
        Return child 1 for the cyclic crossover operator cyclicCrossover().
    
    getCxChild2():
        Returns child 2 for the cyclic crossover operator cyclicCrossover()
        
    cyclicCrossover(p1,p2,ctype):
        Returns tuple (child1,child2). Implements a cyclic crossover operator
        that crosses parents p1 and p2 of chromosome representation type ctype.
        
    orderCrossover(p1,p2,ctype):
        Returns tuple (child1,child2) after crossing parents p1 and p2 using
        order crossover. Chromosome representation selected using ctype.
        
    decodeChromosome2(part_1):
        Returns list containing decoded 'part_1', which is the permutation part
        of a chromosome of type 2.
        
    rationalizeHgaResult(result):
        Returns list containing corrected part_1 of child 2 for the 
        hierarchicalCrossover() operator.
        
    hierarchicalCrossover(p1,p2,C,T,ptype):
        Returns tuple (child1, child2) by crossing parents p1 and p2 using the
        Combined HGA method as proposed in the paper:
            Y. Shuai, S. Yunfeng, and Z. Kai, “An effective method for solving 
            multiple travelling salesman problem based on NSGA-II,” Systems 
            Science & Control Engineering, vol. 7, no. 2, pp. 108–116, Oct. 
            2019, doi: 10.1080/21642583.2019.1674220.
        The first child is based on relative distances from C if problem is 
        MinMax SD-MTSP (ptype==1), else based on the traversal time T. The 
        second child is always based on relative distances.
    
    insertMutation(child,ctype):
        Returns mutated child by selecting a random gene to insert elsewhere,
        selecting a random point (different from original location) and 
        inserting the selected gene there. Can do for either ctype.
        
    swapMutation(child,ctype):
        Returns mutated child by selecting two random genes and swapping their
        positions on chromosome. Can do for either ctype.
        
    invertMutation(child,ctype):
        Returns mutated child by selecting a continuous subset of genes and 
        inverting their ordering. Can do for either ctype.
        
    scrambleMutation(child,ctype):
        Returns mutated child by selecting a continuous subset of genes and
        scrambling the positions of all genes in the subset. Can do for either 
        ctype.
        
    mutateChild(child,ctype):
        Returns mutated child by calling one of the four mutation operators.
        All operators have equal probability of being called. Is the only 
        mutation function to be directly called by external functions.
"""
import numpy as np
import copy
import functools
import src.nsga2 as nsga2
import src.chromosome as chrom

def tournamentSelection(population,tsub=2,selection_probability=1):
    rng = np.random.default_rng()
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

def partiallyMappedCrossover(p1,p2,ctype=2):
    rng = np.random.default_rng()
    if ctype==1:
        c_1,s_1 = p1.cities,p1.tours
        c_2,s_2 = p2.cities,p2.tours
        child_1 = chrom.Chromosome_1()
        child_1.cities = np.zeros(shape=c_1.shape,dtype=int)
        child_1.tours = np.zeros(shape=s_1.shape,dtype=int)
        child_2 = chrom.Chromosome_1()
        child_2.cities = np.zeros(shape=c_2.shape,dtype=int)
        child_2.tours = np.zeros(shape=s_2.shape,dtype=int)
        cut_points = np.sort(rng.choice(c_1.shape[0],2,replace=False))
        for i in range(cut_points[0],cut_points[1]):
            child_2.cities[i],child_2.tours[i] = c_1[i],s_1[i]
            child_1.cities[i],child_1.tours[i] = c_2[i],s_2[i]
        for i in np.concatenate((np.arange(cut_points[0]),
                                 np.arange(cut_points[1],c_1.shape[0]))):
            if c_1[i] not in child_1.cities:
                child_1.cities[i],child_1.tours[i] = c_1[i],s_1[i]
            else:
                child_1.cities[i] = rng.choice([j for j in 
                                                range(1,c_1.shape[0]+1) 
                                                if j not in child_1.cities])
                if np.any(np.arange(1,max(s_1)+1)) not in child_1.tours:
                    child_1.tours[i] = rng.choice([j for j in s_1 
                                                   if j not in child_1.tours])
                else:
                    child_1.tours[i] = rng.choice(np.arange(1,max(s_1)+1))
            if c_2[i] not in child_2.cities:
                child_2.cities[i],child_2.tours[i] = c_2[i],s_2[i]
            else:
                child_2.cities[i] = rng.choice([j for j in 
                                                range(1,c_2.shape[0]+1) 
                                                if j not in child_2.cities])
                if np.any(np.arange(1,max(s_2)+1)) not in child_2.tours:
                    child_2.tours[i] = rng.choice([j for j in s_2 if j 
                                                   not in child_2.tours])
                else:
                    child_2.tours[i] = rng.choice(np.arange(1,max(s_2)+1))
                    
    if ctype==2:
        p11 = p1.part_1
        p21 = p2.part_1
        p12 = p1.part_2
        p22 = p2.part_2
        child_1 = chrom.Chromosome_2()
        child_1.part_1 = np.zeros(shape=p11.shape,dtype=int)
        child_1.part_2 = np.zeros(shape=p12.shape,dtype=int)
        child_2 = chrom.Chromosome_2()
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
                child_1.part_1[i] = rng.choice([j for j in p11 if j 
                                                not in child_1.part_1])
            if child_2.part_1[i] == 0:
                child_2.part_1[i] = rng.choice([j for j in p21 if j 
                                                not in child_2.part_1])
        child_1.part_2 = np.sort(rng.choice(np.arange(1,max(p11)),p12.shape[0],
                                            replace=False))
        child_2.part_2 = np.sort(rng.choice(np.arange(1,max(p21)),p22.shape[0],
                                            replace=False))
    
    
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
          
def cyclicCrossover(p1,p2,ctype=2):
    if ctype==1:
        c_1,s_1 = p1.cities,p1.tours
        c_2,s_2 = p2.cities,p2.tours
        child_1 = chrom.Chromosome_1()
        child_1.cities = np.empty(shape=c_1.shape,dtype=int)
        child_1.tours = np.empty(shape=s_1.shape,dtype=int)
        child_2 = chrom.Chromosome_1()
        child_2.cities = np.empty(shape=c_2.shape,dtype=int)
        child_2.tours = np.empty(shape=s_2.shape,dtype=int)
        start_id = 0
        child_1.cities,child_1.tours = getCxChild1(c_1,s_1,c_2,s_2,
                                                   child_1.cities,
                                                   child_1.tours,start_id)
        child_2.cities,child_2.tours = getCxChild1(c_2,s_2,c_1,s_1,
                                                   child_2.cities,
                                                   child_2.tours,start_id)
        
    if ctype==2:
        rng = np.random.default_rng()
        p11 = p1.part_1
        p21 = p2.part_1
        p12 = p1.part_2
        p22 = p2.part_2
        child_1 = chrom.Chromosome_2()
        child_1.part_1 = np.empty(shape=p11.shape,dtype=int)
        child_1.part_2 = np.empty(shape=p12.shape,dtype=int)
        child_2 = chrom.Chromosome_2()
        child_2.part_1 = np.empty(shape=p21.shape,dtype=int)
        child_2.part_2 = np.empty(shape=p22.shape,dtype=int)
        start_id = 0
        child_1.part_1 = getCxChild2(p11,p21,child_1.part_1,start_id)
        child_2.part_1 = getCxChild2(p21,p11,child_2.part_1,start_id)
        child_1.part_2 = np.sort(rng.choice(np.arange(1,max(p11)),p12.shape[0],
                                            replace=False))
        child_2.part_2 = np.sort(rng.choice(np.arange(1,max(p21)),p22.shape[0],
                                            replace=False))
        
    return (child_1,child_2)

def orderedCrossover(p1,p2,ctype=2):
    rng = np.random.default_rng()
    if ctype==1:
        c_1,s_1 = p1.cities,p1.tours
        c_2,s_2 = p2.cities,p2.tours
        child_1 = chrom.Chromosome_1()
        child_1.cities = np.zeros(shape=c_1.shape,dtype=int)
        child_1.tours = np.zeros(shape=s_1.shape,dtype=int)
        child_2 = chrom.Chromosome_1()
        child_2.cities = np.zeros(shape=c_2.shape,dtype=int)
        child_2.tours = np.zeros(shape=s_2.shape,dtype=int)
        cut_points = np.sort(rng.choice(c_1.shape[0],2,replace=False))
        for i in range(cut_points[0],cut_points[1]):
            child_1.cities[i],child_1.tours[i] = c_1[i],s_1[i]
            child_2.cities[i],child_2.tours[i] = c_2[i],s_2[i]
        remnant_ids = np.concatenate((np.arange(cut_points[1],c_1.shape[0]),
                                      np.arange(cut_points[0])))
        cut_ids = np.concatenate((np.arange(cut_points[1],c_1.shape[0]),
                                  np.arange(cut_points[1])))
        rearr_c1 = [c_1[i] for i in cut_ids]
        rearr_s1 = [s_1[i] for i in cut_ids]
        rearr_c2 = [c_2[i] for i in cut_ids]
        rearr_s2 = [s_2[i] for i in cut_ids]
        rem_c1 = [i for i in rearr_c1 if i not in child_2.cities]
        rem_c2 = [i for i in rearr_c2 if i not in child_1.cities]
        rem_s1 = [rearr_s1[i] for i in range(len(rearr_s1)) if rearr_c1[i] 
                  not in child_2.cities]
        rem_s2 = [rearr_s2[i] for i in range(len(rearr_s2)) if rearr_c2[i] 
                  not in child_1.cities]
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
        child_1 = chrom.Chromosome_2()
        child_1.part_1 = np.zeros(shape=p11.shape,dtype=int)
        child_1.part_2 = np.zeros(shape=p12.shape,dtype=int)
        child_2 = chrom.Chromosome_2()
        child_2.part_1 = np.zeros(shape=p21.shape,dtype=int)
        child_2.part_2 = np.zeros(shape=p22.shape,dtype=int)
        cut_points = np.sort(rng.choice(p11.shape[0],2,replace=False))
        for i in range(cut_points[0],cut_points[1]):
            child_1.part_1[i],child_2.part_1[i] = p11[i],p21[i]
        remnant_ids = np.concatenate((np.arange(cut_points[1],p11.shape[0]),
                                      np.arange(cut_points[0])))
        cut_ids = np.concatenate((np.arange(cut_points[1],p11.shape[0]),
                                  np.arange(cut_points[1])))
        rearr_p11 = [p11[i] for i in cut_ids]
        rearr_p21 = [p21[i] for i in cut_ids]
        rem_p11 = [i for i in rearr_p11 if i not in child_2.part_1]
        rem_p21 = [i for i in rearr_p21 if i not in child_1.part_1]
        j = 0
        for i in remnant_ids:
            child_2.part_1[i],child_1.part_1[i] = rem_p11[j],rem_p21[j]
            j += 1
        child_1.part_2 = np.sort(rng.choice(np.arange(1,max(p11)),p12.shape[0],
                                            replace=False))
        child_2.part_2 = np.sort(rng.choice(np.arange(1,max(p21)),p22.shape[0],
                                            replace=False))
        
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

def hierarchicalCrossover(p1,p2,C,T,ptype=1):
    rng = np.random.default_rng()
    child_1 = chrom.Chromosome_2()
    child_2 = chrom.Chromosome_2()
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
        if ptype==1:
            distances = [C[k,cities[0]],C[k,cities[1]]]
        else:
            distances = [T[k,cities[0]],T[k,cities[1]]]
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
    child_2.part_2 = np.sort(rng.choice(np.arange(1,max(p2.part_1)),
                                        p2.part_2.shape[0],replace=False))
    child_1.part_1 = np.array(result_1)
    child_2.part_1 = np.array(result_2)
    if rng.choice([0,1])==0:
        child_1.part_2 = copy.deepcopy(p1.part_2)
    else:
        child_1.part_2 = copy.deepcopy(p2.part_2)
    return child_1,child_2

def insertMutation(child,ctype=2):
    rng = np.random.default_rng()
    if ctype==1:
        c = child.cities
        s = child.tours
        mutated = chrom.Chromosome_1()
        mutated.cities = copy.deepcopy(c)
        mutated.tours = copy.deepcopy(s)
        point_1 = rng.choice(np.arange(c.shape[0]-1))
        point_2 = rng.choice(np.arange(point_1+1,c.shape[0]))
        mutated.cities = np.insert(mutated.cities,point_1+1,c[point_2])
        mutated.cities = np.delete(mutated.cities,point_2+1)
        
    if ctype==2:
        p1 = child.part_1
        p2 = child.part_2
        mutated = chrom.Chromosome_2()
        mutated.part_1 = copy.deepcopy(p1)
        mutated.part_2 = np.sort(rng.choice(np.arange(1,max(p1)),
                                            p2.shape[0],replace=False))
        point_1 = rng.choice(np.arange(p1.shape[0]-1))
        point_2 = rng.choice(np.arange(point_1+1,p1.shape[0]))
        mutated.part_1 = np.insert(mutated.part_1,point_1+1,p1[point_2])
        mutated.part_1 = np.delete(mutated.part_1,point_2+1)
    
    return mutated
  
def swapMutation(child,ctype=2):
    rng = np.random.default_rng()
    if ctype==1:
        c = child.cities
        s = child.tours
        mutated = chrom.Chromosome_1()
        mutated.cities = copy.deepcopy(c)
        mutated.tours = copy.deepcopy(s)
        diff_tour_pairs = [[i,j] for i in range(c.shape[0]) 
                           for j in range(c.shape[0]) if s[i]!=s[j]]
        if diff_tour_pairs == []:
            mutated = child
            return mutated
        points = rng.choice(diff_tour_pairs)
        mutated.cities[points[0]],mutated.cities[points[1]] = mutated.cities[
            points[1]],mutated.cities[points[0]]
        mutated.tours[points[0]],mutated.tours[points[1]] = mutated.tours[
            points[1]],mutated.tours[points[0]]
        
    if ctype==2:
        p1 = child.part_1
        p2 = child.part_2
        mutated = chrom.Chromosome_2()
        mutated.part_1 = copy.deepcopy(p1)
        mutated.part_2 = np.sort(rng.choice(np.arange(1,max(p1)),p2.shape[0],
                                            replace=False))
        points = rng.choice(np.arange(p1.shape[0]),2,replace=False)
        mutated.part_1[points[0]],mutated.part_1[points[1]] = mutated.part_1[
            points[1]],mutated.part_1[points[0]]
    
    return mutated
  
def invertMutation(child,ctype=2):
    rng = np.random.default_rng()
    if ctype==1:
        c = child.cities
        s = child.tours
        mutated = chrom.Chromosome_1()
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
        mutated = chrom.Chromosome_2()
        mutated.part_1 = copy.deepcopy(p1)
        mutated.part_2 = np.sort(rng.choice(np.arange(1,max(p1)),p2.shape[0],
                                            replace=False))
        points = np.sort(rng.choice(np.arange(p1.shape[0]),2,replace=False))
        inverse_p1 = [p1[i] for i in range(points[1],points[0]-1,-1)]
        j = 0
        for i in range(points[1],points[0]-1,-1):
            mutated.part_1[i] = inverse_p1[j]
            j += 1
    
    return mutated
  
def scrambleMutation(child,ctype=2):
    rng = np.random.default_rng()
    if ctype==1:
        c = child.cities
        s = child.tours
        mutated = chrom.Chromosome_1()
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
        mutated = chrom.Chromosome_2()
        mutated.part_1 = copy.deepcopy(p1)
        mutated.part_2 = np.sort(rng.choice(np.arange(1,max(p1)),p2.shape[0],
                                            replace=False))
        points = np.sort(rng.choice(np.arange(p1.shape[0]),2,replace=False))
        scramble_ids = rng.permutation(np.arange(points[0],points[1]+1))
        scrambled_p1 = [p1[i] for i in scramble_ids]
        for i,scrambled_id in enumerate(scramble_ids):
            mutated.part_1[scrambled_id] = scrambled_p1[i]
    
    return mutated

def mutateChild(child,ctype=2):
    rng = np.random.default_rng()
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

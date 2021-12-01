def main():
    n_iters = 1000
    n_cities = 51
    n_tours = 7
    map_size = 500
    ctype = 2
    pop_size = 100
    tsub = 2
    selection_probability = 1
    cx_type = 'hx'
    mutation_probability = 0.05
    instance_type = 'from file'
    instance_file_name = 'eil51.txt'
    
    print("\n-------------------- PROGRAM START --------------------\n")
    print("*** Multi-Objective Problem: MOmTSP with",n_cities,"cities and",
          n_tours,"salespersons ***\n",
          "                    *** Solving Algorithm: NSGA_II *** ")
    if instance_type=='random':
        print("Generating Instance...")
        C = generateInstance(n_cities,map_size)
    else:
        print("Reading instance from file:",instance_file_name,"...")
        C,data = readInstance(instance_file_name)
    
    cx_all = ['hx']
    mut_range = [0.05]
    alg_type = 'original'
    exp_no = 0
    front_dict = dict()
    plt.figure()
    script_dir = os.path.dirname(__file__)
    results_dir_1 = os.path.join(script_dir, 
                              "For_Video/{!r}".format(instance_file_name[:-4]))
    results_dir_2 = os.path.join(script_dir, "Results/data/")
    results_dir_3 = os.path.join(script_dir, "Results/figures/")
    if not os.path.isdir(results_dir_1):
        os.makedirs(results_dir_1)
    if not os.path.isdir(results_dir_2):
        os.makedirs(results_dir_2)
    if not os.path.isdir(results_dir_3):
        os.makedirs(results_dir_3)
    #-----------------MAIN LOOP-----------------
    for cx_type in cx_all:
        for mutation_probability in mut_range:
            exp_no += 1
            print(">>>EXPERIMENT NO.",exp_no,"")
            print("PARAMETERS: \nCrossover type->",cx_type,
                  "; Mutation Probability->",mutation_probability,
                  "; n(iterations)->",n_iters)
            print("Population size->",pop_size,"; original/modified->",
                  alg_type)
            print("\nCreating initial population...\n")
            population = createInitialPopulation(pop_size,C,data,n_tours,ctype)
            extra_front = []
            first_front = []
            print("   >>>Entering Main Loop:\n")
            for iter_count in tqdm.tqdm(range(n_iters)):
            #for iter_count in range(n_iters):
                fronts = nondominatedSort(population,ctype)
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
                        extra_front = sorted(extra_front,
                                             key=functools.cmp_to_key(
                                                 crowdedComparisonOperator))
                    next_generation_P.extend(extra_front[0:pop_size-P_temp_length])
                next_generation_Q = createOffspringPopulation(next_generation_P,C,
                                                              tsub,
                                                              selection_probability,
                                                              cx_type,
                                                              mutation_probability,
                                                              ctype)
                population = next_generation_P
                population.extend(next_generation_Q)
                #plt.clf()
                first_front = fronts[0]
                #fvalues = [(i.function_vals[0],i.function_vals[1]) 
                #           for i in first_front]
                #X = [-i[0] for i in fvalues]
                #Y = [-i[1] for i in fvalues]
                #plt.figure()
                #plt.xlabel("Total Distance")
                #plt.ylabel("Average Tour Distance")
                #plt.scatter(X,Y)
                #plt.show()
            print("\n   >>>Finished Main Loop")
            fvalues1 = [(i.function_vals[0],i.function_vals[1]) 
                        for i in first_front]
            X = [-i[0] for i in fvalues1]
            Y = [-i[1] for i in fvalues1]
            
            plt.xlabel("Total Distance")
            plt.ylabel("Average Tour Distance")
            #plt.title("Exp_no: {!r}; cross: {!r}; mut: {!r}".format(exp_no,
             #                                           cx_type,
              #                                          mutation_probability))
            plt.scatter(X,Y)
            
            front_dict[exp_no] = {"cx":cx_type,"mu":mutation_probability,
                                  "front":first_front}
            plt.savefig(results_dir_3+
                        "Exp_{!r}_ctype_{!r}_instance_{!r}".format(exp_no,
                                              cx_type,instance_file_name[:-4]))
            #save front and the parameters of the experiment in dict
            #save figures as png in folder results with name having
            #experiment parameters
            with open(
                "Results/data/Exp_{!r}_ctype_{!r}_instance_{!r}.pkl".format(
                    exp_no,cx_type,instance_file_name[:-4]),"wb") as wrfile:
                pickle.dump(front_dict,wrfile)
    print("--------------- PROGRAM END ---------------")
    return 0
        
if __name__=="__main__":
    main()

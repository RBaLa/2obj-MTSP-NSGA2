from src import utils

def main():
    #VARIABLES:
    #-------------------------------------------------------------------------------------------------------
    number_of_iterations = 1000
    number_of_tours = 7         # Note: is the number of salesmen
    population_size = 100
    selection_probability = 1   # Options: 0<selection_probability<=1
    crossover_type = 'hx'       # Options:  'pmx':partially-mapped,
                                #           'cycx':cyclic,
                                #           'ox':ordered,
                                #           'hx':heirarchical
    mutation_probability = 0.05 # Options: 0<mutation_probability<=1
    chromosome_type = 2         # Options:  1:Two-string chromosome, 
                                #           2:Single-string breakpoint-type chromosome
    #***WARNING*** If chromosome_type == 1, 'hx' should not be chosen, otherwise will throw error. 
    #For random instance generation:
    number_of_cities = 51  # Note: including depot city
    map_size = 500         # Note: the map size is map_size x map_size
    #-------------------------------------------------------------------------------------------------
    
    instance_type,instance_file = utils.getInstanceFromUser()
    save_fd = input("Save plot and final generation fronts?(y/n) [default:n]:")
    if save_fd=='y':
        script_dir = os.path.dirname(__file__)
        results_dir_1 = os.path.join(script_dir, "Results/data/")
        results_dir_2 = os.path.join(script_dir, "Results/figures/")
        if not os.path.isdir(results_dir_1):
            os.makedirs(results_dir_1)
        if not os.path.isdir(results_dir_2):
            os.makedirs(results_dir_2)
    
    if instance_type=='random':
        print("Generating Instance...")
        C = generateInstance(n_cities,map_size)
    else:
        print("Reading instance from file:",instance_file,"...")
        C,data = readInstance(instance_file)
        
    front_dict = dict()
    plt.figure()
    
    print("PARAMETERS: \nCrossover type->",cx_type,
                  "; Mutation Probability->",mutation_probability,
                  "; n(iterations)->",n_iters)
    print("Population size->",pop_size)
    print("\nCreating initial population...\n")
    population = createInitialPopulation(population_size,C,data,number_of_tours,ctype)
    extra_front = []
    first_front = []
    for iter_count in tqdm.tqdm(range(number_of_iterations)):
        
    
    
    return 0
        
if __name__=="__main__":
    main()

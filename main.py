def getInstanceFromUser():
    try_count = 0
    while True:
        try:
            value = str(input("Read instance from file?(y/n) [default:y. If n is selected, random instance will be generated]:"))
        except ValueError:
            print("Error: something other than 'y' or 'n' entered. Try again.")
            try_count+=1
            if try_count==10:
                sys.exit(1)
            continue
        if value=='y' or value=='' or (!value.isalpha() and value.isspace()):
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            script_dir = os.path.dirname(__file__)
            instance_dir = os.path.join(script_dir, "instances")
            instance_file = filedialog.askopenfilename(initialdir=instance_dir, title="Select file",
                                                       filetypes=[("Text Files", "*.txt")])
            if len(instance_file)==0:
                proceed_quit = str(input("No file selected. Proceed to random instance generation or quit?(y/q) [default:y]"))
                if proceed_quit=='y' or proceed_quit=='' or (!proceed_quit.isalpha() and proceed_quit.isspace()):
                    instance_type = 'random'
                    instance_file = None
                    break
            else:
                instance_type = 'from file'
                break
        elif value=='n':
            proceed_quit = str(input("No selected. Proceed to random instance generation or quit?(y/q) [default:y]"))
            if proceed_quit=='y' or proceed_quit=='' or (!proceed_quit.isalpha() and proceed_quit.isspace()):
                instance_type = 'random'
                instance_file = None
                break
        else:
            print("Error: something other than 'y' or 'n' entered. Try again.")
            try_count+=1
            if try_count==10:
                sys.exit(1)
            continue
    return instance_type,instance_file

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
    
    instance_type,instance_file = getInstanceFromUser()
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

# -*- coding: utf-8 -*-
"""
main.py (script)

Written as part of the submission for ECE 750 AL: Bio&Comp Individual Project
Author: Rahul Balamurugan

Contains the script that controls all the functions and contains input 
parameters. Runs the NSGA-II program if executed.
"""

import src.utils as utils
import src.evolution as evo

def main():
    #VARIABLES:
    #-------------------------------------------------------------------------
    number_of_repetitions = 3
    number_of_iterations = 100
    problem_variation = 1       # Options: 1:bi-objective MinMax SD-MTSP
                                #          2:bi-objective SD-MTSP 
                                #           with objectives->minimize total
                                #           cost, minimize sum of differences
                                #           between individual tour times and
                                #           avg. tour time.
    number_of_tours = 7         # Note: is the number of salesmen
    population_size = 100
    selection_probability = 1   # Options: 0<selection_probability<=1
    crossover_type = 'hx'       # Options:  'pmx':partially-mapped,
                                #           'cycx':cyclic,
                                #           'ox':ordered,
                                #           'hx':heirarchical
    mutation_probability = 0.05 # Options: 0<mutation_probability<=1
    tournament_bracket_size = 2 # Note: is number of competing individuals
                                #       in tournament round (2=>Binary)
    chromosome_type = 2         # Options:  1:Two-string chromosome, 
                                #           2:Single-string breakpoint-type 
                                #             chromosome (preferred)
    
    #***WARNING*** If chromosome_type == 1, 'hx' should not be chosen, 
    #otherwise program will throw error. ***WARNING finished***
    
    #For random instance generation:
    number_of_cities = 51  # Note: including depot city at (0,0) RED on plot
    map_size = 500         # Note: the map size is map_size x map_size
    #-------------------------------------------------------------------------
    
    instance_type,instance_file = utils.getInstanceFromUser()
    save_fd = input("Save plot and final generation fronts?(y/n) [default:n]:")
    
    if instance_type=='random':
        print("Generating Instance...")
        C,data,T = utils.generateInstance(number_of_cities,map_size)
    else:
        print("Reading instance from file:",instance_file,"...")
        C,data,T = utils.readInstance(instance_file)
        number_of_cities = C.shape[0]
    
    print("\n-------------------- PROGRAM START --------------------\n")
    print("*** Multi-Objective Problem: MOmTSP with",number_of_cities,
          "cities and",number_of_tours,"salespersons ***\n")
    
    print("PARAMETERS: \nCrossover type->",crossover_type,
                  "; Mutation Probability->",mutation_probability,
                  "; n(iterations)->",number_of_iterations)
    print("Population size->",population_size)
    
    best_front,fronts = evo.evolve(population_size,C,T,data,number_of_tours,
                            number_of_iterations,crossover_type,
                            chromosome_type,selection_probability,
                            mutation_probability,tournament_bracket_size,
                            number_of_repetitions)
    if save_fd=='y':
        utils.saveData(fronts)
        utils.plotAndSaveFigures(best_front,population_size,problem_variation,
                                 'y')
    else:
        utils.plotAndSaveFigures(best_front,population_size,problem_variation)
    print("\n--------------------- PROGRAM END ---------------------\n")
    
if __name__=="__main__":
    main()
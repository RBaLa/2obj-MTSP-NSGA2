# -*- coding: utf-8 -*-
"""
utils.py (module)

Written as part of the submission for ECE 750 AL: Bio&Comp Individual Project
Author: Rahul Balamurugan

Contains general utility functions for the program. Does nothing if executed.

List of functions:
    
    getTourMatrix(chromosome,ctype=1):
        Returns matrix X of size n*n*m with elements X[i,j,k]=1 iff salesman k
        in 'chromosome' (of type 'ctype') traverses the arc between city i 
        and city j.
    
    euclideanDistance(a,b):
        Returns 2D euclidean distance as a floating point number between points
        a and b, both argument variables being lists of length 2.
    
    generateInstance(n_cities,map_size):
        Returns tuple of (C,coords,T). The function randomly distributes 
        'n_cities' number of points on 2D square region with opposite
        vertices at (0,0) and ('map_size','map_size'), each point stored in
        list 'coords'. The distance matrix 'C' is computed between all the
        points in 'coord'. Each value in 'C' is divided by a random value 
        (speed) between 20 km/hr and 90 km/hr (rand(20000,90001) when C in 'm')
        to get time matrix T.
    
    readInstance(filename):
        Returns tuple of (C,coords,T). Reads the text file 'filename' 
        containing a TSP instance in format prescribed in TSPLIB, and stores
        city coordinates in 'coord'. Computes distance matrix between all
        cities, 'C', and computes 'T' from 'C' in the same way as 
        generateInstance().
        
    getInstanceFromUser():
        Waits for and gets user input of instance file type (random/from file)
        and if the latter, also returns the file name. Allows user to browse
        the directory for instance text files.
        
    plotAndSaveFigures(best_front,pop_size,ptype=1,saverno='n'):
        Plots function values of individuals in best_front. Saves the figure
        as both .png and .svg files in Results folder.
        
    saveFronts(fronts):
        Saves fronts as pickle file in Results folder.
    
"""
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import functools
import pickle
from tkinter import Tk,filedialog
import src.nsga2 as nsga2



def getTourMatrix(chromosome,ctype=1):
    if ctype==1:
        n_tours = max(chromosome.tours) #same as no. of salespersons
        n_cities = max(chromosome.cities)
        tour_matrix = np.zeros(shape=(n_tours,n_cities+1,n_cities+1),dtype=int)
        tour_order_lists = [[] for i in range(n_tours)]
        for i in range(n_cities):
            tour_order_lists[chromosome.tours[i]-1].append(
                chromosome.cities[i])
        
    if ctype==2:
        part_1 = chromosome.part_1
        part_2 = chromosome.part_2
        n_tours = part_2.shape[0]+1 #again, this is no. of salespersons
        n_cities = part_1.shape[0]
        tour_matrix = np.zeros(shape=(n_tours,n_cities+1,n_cities+1),dtype=int)
        tour_order_lists = [[] for i in range(n_tours)]
        count = 0
        breakpoints = [part_2[i] for i in range(n_tours-1)]
        breakpoints.append(n_cities)
        for i in range(n_tours):
            breakpoint = breakpoints[i]
            for j in range(count,breakpoint):
                tour_order_lists[i].append(part_1[j])
            count = copy.deepcopy(breakpoint)
    
    for i,tours in enumerate(tour_order_lists):
        for j,city in enumerate(tours):
            if j!=0:
                tour_matrix[i,tours[j-1],city] = 1
            else:
                tour_matrix[i,0,city] = 1
            if j==len(tours)-1:
                tour_matrix[i,city,0] = 1
    return tour_matrix

def euclideanDistance(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

def generateInstance(n_cities,map_size):
    rng = np.random.default_rng()
    city_coordinates = [(0,0)] #initialized with depot
    distance_matrix = np.empty(shape=(n_cities,n_cities),dtype=float)
    while len(city_coordinates)<n_cities:
        x_choice = rng.choice(map_size)
        y_choice = rng.choice(map_size)
        new_city = (x_choice,y_choice)
        if new_city not in city_coordinates: #ensuring only new entries
            city_coordinates.append(new_city)
    for i,city_i in enumerate(city_coordinates):
        for j,city_j in enumerate(city_coordinates):
            distance_matrix[i,j] = euclideanDistance(city_i,city_j)
    #Plot generate instance:
    plt.figure(figsize=(8,8))
    x = [p[0] for p in city_coordinates]
    y = [p[1] for p in city_coordinates]
    plt.xlim([-map_size/100,map_size+map_size/100])
    plt.ylim([-map_size/100,map_size+map_size/100])
    plt.plot(x,y,"ob")
    plt.plot(0,0,"or")
    plt.grid()
    plt.show()
    time_matrix = np.empty(shape=(len(city_coordinates),
                                      len(city_coordinates)),dtype=int)
    for i,city_i in enumerate(city_coordinates):
        for j,city_j in enumerate(city_coordinates):
            time_matrix[i,j] = distance_matrix[i,j]/rng.integers(low=20000,high=90001)
    return distance_matrix,city_coordinates,time_matrix

def readInstance(filename):
    city_coordinates = []
    with open(filename) as f:
        for line in f.readlines():
            if line[0].isnumeric():
                int_list = [int(float(i)) for i in line.split()]
                city_coordinates.append((int_list[1],int_list[2]))
    distance_matrix = np.empty(shape=(len(city_coordinates),
                                      len(city_coordinates)),dtype=int)
    for i,city_i in enumerate(city_coordinates):
        for j,city_j in enumerate(city_coordinates):
            distance_matrix[i,j] = round(euclideanDistance(city_i,city_j))
    #Plot generate instance:
    plt.figure()
    x = [p[0] for p in city_coordinates]
    y = [p[1] for p in city_coordinates]
    plt.plot(x,y,"ob")
    plt.plot(city_coordinates[0][0],city_coordinates[0][1],"or")
    plt.grid()
    plt.show()
    rng = np.random.default_rng()
    time_matrix = np.empty(shape=(len(city_coordinates),
                                      len(city_coordinates)),dtype=float)
    for i,city_i in enumerate(city_coordinates):
        for j,city_j in enumerate(city_coordinates):
            time_matrix[i,j] = distance_matrix[i,j]/rng.integers(low=20000,high=90001)
    return distance_matrix,city_coordinates,time_matrix

def getInstanceFromUser():
    try_count = 0
    while True:
        try:
            value = str(input("Read instance from file?(y/n) [default:y. If n is selected, random instance will be generated]:"))
        except ValueError:
            print("Error: something other than 'y' or 'n' entered. Try again.")
            try_count+=1
            if try_count==10:
                print("Error: Too many wrong attempts. Re-run the program.")
                sys.exit(1)
            continue
        if value=='y' or value=='' or (not(value.isalpha()) and value.isspace()):
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            script_dir = os.path.dirname(__file__)
            instance_dir = os.path.join(script_dir, "instances")
            instance_file = filedialog.askopenfilename(initialdir=instance_dir, title="Select file",
                                                       filetypes=[("Text Files", "*.txt")])
            if len(instance_file)==0:
                proceed_quit = str(input("No file selected. Proceed to random instance generation or quit?(y/q) [default:y]:"))
                if proceed_quit=='y' or proceed_quit=='' or (not(proceed_quit.isalpha()) and proceed_quit.isspace()):
                    instance_type = 'random'
                    instance_file = None
                    break
                else:
                    sys.exit(0)
            else:
                instance_type = 'from file'
                break
        elif value=='n':
            proceed_quit = str(input("No selected. Proceed to random instance generation or quit?(y/q) [default:y]"))
            if proceed_quit=='y' or proceed_quit=='' or (not(proceed_quit.isalpha()) and proceed_quit.isspace()):
                instance_type = 'random'
                instance_file = None
                break
            else:sys.exit(0)
        else:
            proceed_quit = str(input("Error: something other than 'y' or 'n' entered. Try again or quit?(y/q)[default:y]:."))
            if proceed_quit=='q':
                sys.exit(0)
            try_count+=1
            if try_count==10:
                print("Error: Too many wrong attempts. Re-run the program.")
                sys.exit(1)
            continue
    return instance_type,instance_file

def plotAndSaveFigures(best_front,pop_size,ptype=1,saverno='n'):
    
    plt.figure()
    plt.xlabel("Total cost (distance)")
    if ptype==1:
        plt.ylabel("Max-min tour distances (amplitude)")
    else:
        plt.ylabel("Sum(individual tour time-avg. tour time)")
    final_front = nsga2.nondominatedSort(best_front, 2)
    bestest_front = nsga2.assignCrowdingDistance(final_front[0])
    bestest_front = sorted(bestest_front,key=functools.cmp_to_key(
                                            nsga2.crowdedComparisonOperator))
    fvalues1 = [(i.function_vals[0],i.function_vals[1]) 
                    for i in bestest_front[:pop_size]]
    X = [-i[0] for i in fvalues1]
    Y = [-i[1] for i in fvalues1]
    plt.plot(X,Y,'og')
    
    if saverno=='y':
        script_dir = os.getcwd()
        results_dir_2 = os.path.join(script_dir, "Results/figures/")
        if not os.path.isdir(results_dir_2):
            os.makedirs(results_dir_2)
        plt.savefig(results_dir_2+"opt_pareto_front.png")
        plt.savefig(results_dir_2+"opt_pareto_front.svg")
        
    plt.show()
    
def saveData(fronts):
    script_dir = os.getcwd()
    results_dir_1 = os.path.join(script_dir, "Results/data/")
    if not os.path.isdir(results_dir_1):
        os.makedirs(results_dir_1)
    with open(results_dir_1+"all_fronts.pkl","wb") as wrfile:
        pickle.dump(fronts,wrfile)
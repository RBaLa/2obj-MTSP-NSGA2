import numpy as np
import matplotlib.pyplot as plt
import seed
from tkinter import Tk,filedialog
import os

def updateSeed():
    large = 2147483647;
    k = int(seed.SEED/127773);
    seed.SEED = 16807*(seed.SEED-k*127773)-k*2836;
    if seed.SEED<=0:
        seed.SEED += large;

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
            if j==len(tours):
                tour_matrix[i,city,0] = 1
    return tour_matrix
  
  def euclideanDistance(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
  
def generateInstance(n_cities,map_size):
    rng = np.random.default_rng(seed.SEED)
    updateSeed()
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
    return distance_matrix,city_coordinates
  
def readInstance(filename):
    city_coordinates = []
    with open(filename) as f:
        for line in f.readlines():
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
    return distance_matrix,city_coordinates

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


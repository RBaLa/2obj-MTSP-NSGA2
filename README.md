# Multi-objective optimization of the Multiple Traveling Salesmen Problem Using a Non-dominated Sorting Genetic Algorithm (NSGA-II)
Deliverable for the ECE 750 Artificial Life: Biology and Computation Fall 2021 course @ UWaterloo
### Author: Rahul Balamurugan

The NSGA-II algorithm is implemented faithful to the original as far as the non-dominated sorting, crowding distance operator, and binary crowded tournament selection go. However, the crossover and mutation operators are those meant for permutation based problems.

Available crossover operators: Partially-mapped crossover (PMX), Cyclic crossover (CX), Ordered crossover (OX), and heirarchical crossover (HX)(default). The HX operator is based on the proposed algorithm from the paper "An effective method for solving multiple travelling salesman problem based on NSGA-II" (2019) by Yang Shuai, Shao Yunfeng & Zhang Kai. Available at https://doi.org/10.1080/21642583.2019.1674220.

Implemented mutation operators: Insert mutation, Swap mutation, Invert mutation, and Scramble mutation. All four have an equal chance to be chosen if the child chromosome is selected for mutation.

Two objective functions  for the MTSP: minimizing total distance traveled and two separate options for the second, either minimizing difference between max and min tour length (MinMax SD-MTSP) or minimizing difference between the maximum time taken for a tour and average across all salesmen. For the latter option, a random speed between 40 and 90 km/h is chosen for each arc traversal.

Available instances from TSPLIB: eil51, berlin52, eil76, and rat99.

### Instructions to run:
1. Download code as zipped file.
2. Unzip and open main.py in an editor (if you want to change parameters)
3. Inside main.py, change variables as desired in function main()
4. Run main.py and select instance when prompted ('random' for random instance generation)
5. Also select 'yes' or 'no' for saving the image and obtained fronts when prompted
6. If 'yes' was selected (default), results will be in a folder named 'Results' which will be created automatically if not present in the working directory. Results include .png figure of the optimal pareto front and the .pkl file containing the dictionary of fronts obtained in last generation.

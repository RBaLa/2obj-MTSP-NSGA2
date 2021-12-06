# -*- coding: utf-8 -*-
"""
mtsp.py (module)

Written as part of the submission for ECE 750 AL: Bio&Comp Individual Project
Author: Rahul Balamurugan

Contains all functions that describe the two problem variations of the MTSP,
the MinMax SD-MTSP and another SD-MTSP with additional objective of 
balancing the individual tour durations. Does nothing if executed.

List of contained functions:
    
    objectiveFunction1(C,X):
        Returns total distance travelled by all salesmen. X is traversal matrix
        of shape (i*j*k), that is 1 iff salesman k traverses arc between city
        i and city j. C is distance matrix, with each element C[i,j] being
        the euclidean distance between city i and city j. Diagonal is filled
        with zeroes.
    
    objectiveFunction2(C,X,ftype):
        Returns difference between max tour length and min tour length if 
        ftype=='r', else if ftype=='td', returns sum of time difference
        between each tour and the average tour time. If ftype=='std' returns 
        the standard deviation of all tour distances (not used). 
        
"""

import numpy as np

def objectiveFunction1(C,X):
    total_distance = 0
    for i,tour in enumerate(X):
        distance = np.sum(np.multiply(C,tour))
        total_distance += distance
    return (-total_distance)

def objectiveFunction2(C,T,X,ftype=1):
    n_tours = X.shape[0]
    tour_lengths = np.empty(shape=(n_tours,),dtype=float)
    tour_times = np.empty(shape=(n_tours,),dtype=float)
    total_distance = 0
    total_time = 0
    avg_tour_length = 0
    variation = 0
    
    if ftype==2: #Total deviation from average time
        for i,tour in enumerate(X):
            tourtime = np.sum(np.multiply(T,tour))
            total_time += tourtime
            tour_times[i] = tourtime
        avg_tour_time = total_time/n_tours
        for i in range(n_tours):
            variation += np.abs(avg_tour_time - tour_times[i])
        return (-variation)*60
    
    if ftype==1: #Range of tour lengths
        for i,tour in enumerate(X):
            tour_lengths[i] = np.sum(np.multiply(C,tour))
        return -(max(tour_lengths)-min(tour_lengths))
    
    if ftype=='std': #Standard deviation
        for i,tour in enumerate(X):
            distance = np.sum(np.multiply(C,tour))
            total_distance += distance
            tour_lengths[i] = distance
        avg_tour_length = total_distance/n_tours
        for i in range(n_tours):
            variation += (avg_tour_length - tour_lengths[i])**2
        return -((variation/(n_tours-1))**0.5)
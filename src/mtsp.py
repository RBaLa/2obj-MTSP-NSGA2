def objectiveFunction1(C,X):
    total_distance = 0
    for i,tour in enumerate(X):
        distance = np.sum(np.multiply(C,tour))
        total_distance += distance
    return (-total_distance)

def objectiveFunction2(C,X,ftype='r'):
    n_tours = X.shape[0]
    tour_lengths = np.empty(shape=(n_tours,),dtype=float)
    total_distance = 0
    avg_tour_length = 0
    variation = 0
    
    if ftype=='td': #Total deviation from average
        for i,tour in enumerate(X):
            distance = np.sum(np.multiply(C,tour))
            total_distance += distance
            tour_lengths[i] = distance
        avg_tour_length = total_distance/n_tours
        for i in range(n_tours):
            variation += np.abs(avg_tour_length - tour_lengths[i])
        return (-variation)
    
    if ftype=='r': #Range of tour lengths
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

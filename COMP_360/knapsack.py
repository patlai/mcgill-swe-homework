def knapsack(values, weights, maxWeight):
    """
    Dynamic programming algorithm for solving the 0-1 knapsack problem
    """

    # remove all items higher than max weight from consideration
    canFit = np.array(weights) < maxWeight
    values = [values[i] for i in range (0, len(values)) if canFit[i]]
    weights = [weights[i] for i in range (0, len(weights)) if canFit[i]]
    
    n = len(values)
    A = np.zeros([n, maxWeight + 1])
    A[0, weights[0]] = values[0]
    
    for i in range(1, n):
        for w in range(0, maxWeight + 1):
            # if there is a set from items[0, i-1] of weight exactly w, then
            # there must be a set from items[0, i] of at least equal value
            A[i, w] = A[i-1, w] if A[i-1, w] > 0 else A[i, w]
        for w in range (weights[i], maxWeight + 1):
            # if there is a set of items [0, i-1] that adds up to weight w-w_i,
            # and adding item i would yield a greater value than the current subset of
            # items [0, i], then add item i to the set of items to keep
            if A[i-1, w-weights[i]] > 0 and A[i-1, w-weights[i]] + values[i] > A[i, w]:
                A[i, w] = A[i-1, w-weights[i]] + values[i]
                
    return np.max(A)
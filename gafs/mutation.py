import numpy as npy

def mutation(individual, mutation_rate = 0.1, mutation_preference = 'less'):
    """
    APPLY MUTATION WITH BIAS TOWATDS SELECTING FEWER OR MORE VARIABLES

    Parameters
    ----------
    individual : list
        Binary representation of a feature subset.
    mutation_rate : float
        Probability of mutating each gene.
    mutation_preference : str, optional
        - 'less' : Prioritizes turning 1 → 0 (favoring fewer features).
        - 'more' : Prioritizes turning 0 → 1 (favoring more features).
        - 'neutral' : Standard mutation (equal probability for 0 → 1 and 1 → 0).

    Returns
    -------
    list
        Mutated individual.
    """
    
    for i in range(len(individual)):
        
        if npy.random.rand() < mutation_rate:
            
            if mutation_preference == "less" and individual[i] == 1:  # Prioritizes 1 → 0
                individual[i] = 0
            elif mutation_preference == "more" and individual[i] == 0:  # Prioritizes 0 → 1
                individual[i] = 1
            elif mutation_preference == "neutral":  # Standard Mutation
                individual[i] = 1 - individual[i]
                
    return individual


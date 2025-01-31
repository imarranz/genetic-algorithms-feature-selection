import numpy as npy

def roulette_selection(population, scores):
    """
    SELECT INDIVIDUALS FROM THE POPULATION BASED ON THEIR SCORES USING A ROULETTE WHEEL METHODOLOGY

    Parameters
    ----------
    population : list of lists
        The current population of individuals.
    scores : list of floats
        The fitness scores corresponding to the individuals in the population.

    Returns
    -------
    list of lists
        The selected individuals.
    """

    # Calculate the total sum of scores
    total = sum(scores)

    # Calculate the probabilities of selection based on scores
    probabilities = [s / total for s in scores]

    # Randomly select indices from the population with probabilities according to scores
    indices = npy.random.choice(len(population), size = len(population), p = probabilities)

    # Return a new population based on the selected indices
    return [population[i] for i in indices]

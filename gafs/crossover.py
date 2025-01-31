import numpy as npy

def crossover(parent1, parent2):
    """
    PERFORM CROSSOVER BETWEEN TWO PARENT INDIVIDUALS TO PRODUCE TWO OFFSPRING

    This function implements a single-point crossover operation. A random
    crossover point is selected, and the offspring are created by exchanging
    the genetic material of the parents at that point.

    Parameters
    ----------
    parent1 : list of int
        A binary list representing the first parent individual.
    parent2 : list of int
        A binary list representing the second parent individual.

    Returns
    -------
    tuple of list of int
        Two offspring individuals produced by the crossover.

    Notes
    -----
    - The crossover point is randomly chosen between the first and last positions of the parents.
    - This operation ensures diversity in the new population by combining features from both parents.
    """

    # Generate a random point for crossover
    point = npy.random.randint(1, len(parent1))
    
    # Create children by combining parts of both parents based on the crossover point
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2

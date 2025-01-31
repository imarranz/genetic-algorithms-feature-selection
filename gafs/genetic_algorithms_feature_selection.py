import pandas as pd
import numpy as npy
from sklearn.model_selection import cross_val_score

import os
from datetime import datetime
from tqdm import tqdm
import joblib

from .evaluation import evaluate
from .selection import roulette_selection
from .crossover import crossover
from .mutation import mutation
from .plot import plot_gafs_results

def genetic_algorithms_feature_selection(
    X,
    y,
    model,
    metric = 'accuracy',
    generations = 50,
    population_size = 20,
    mutation_rate = 0.1,
    mutation_preference = 'neutral',
    penalty = 0.01,
    random_state = None,
    save_results = False,
    population = None  # Nuevo argumento para población inicial
):
    """
    PERFORM FEATURE SELECTION USING GENETIC ALGORITHMS (GAFS)

    This function implements a Genetic Algorithm (GA) to identify the best subset
    of features for a classification problem. It evaluates feature subsets based
    on the provided model and evaluation metric, while also penalizing large subsets.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix containing the data.
    y : ndarray of shape (n_samples,)
        Target vector containing class labels.
    model : object
        A scikit-learn-compatible classifier (e.g., RandomForestClassifier, LogisticRegression).
    metric : str, optional
        Metric used for model evaluation during cross-validation. Default is 'accuracy'.
        Examples: 'accuracy', 'roc_auc', 'f1', etc.
    generations : int, optional
        Number of generations for the genetic algorithm. Default is 50.
    population_size : int, optional
        Number of individuals in the population. Default is 20.
    mutation_rate : float, optional
        Probability of mutation for each gene in an individual. Default is 0.1.
    mutation_preference : str, default 'neutral'.
        - 'less' : Prioritizes turning 1 → 0 (favoring fewer features).
        - 'more' : Prioritizes turning 0 → 1 (favoring more features).
        - 'neutral' : Standard mutation (equal probability for 0 → 1 and 1 → 0).
    penalty : float, optional
        Penalty factor for the number of selected features. Higher values penalize larger subsets. Default is 0.01.
    random_state : int, optional
        Seed for reproducibility. Default is None.
    save_results : bool, optional
        If True, saves the results to a joblib file with the timestamp as filename. Default is False.
    population : list or None, optional
        If None, initializes a random population. If provided, it must be a list of binary lists 
        (each representing a feature subset). Default is None.


    Returns
    -------
    dict
        A dictionary containing:
        - 'best_subset' : list
            Binary representation of the best feature subset.
        - 'best_score' : float
            Best evaluation score achieved.
        - 'selected_features' : int
            Number of selected features in the best subset.
        - 'evolution' : pandas.DataFrame
            DataFrame showing the best score at each generation.
        - 'logs' : list
            List of log messages recording events during the execution.
        - 'parameters' : dict
            Dictionary with the parameters used for the function call.

    Notes
    -----
    - This function uses Genetic Algorithms (GAs) to optimize feature selection.
    - The evaluation process applies cross-validation with the specified metric.
    - Penalizing larger feature subsets encourages simpler models and helps prevent overfitting.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X, y = make_classification(n_samples = 100, n_features = 10, random_state = 42)
    >>> model = RandomForestClassifier(random_state = 42)
    >>> result = gafs_feature_selection(
    ...     X, y,
    ...     model = model,
    ...     metric = 'accuracy',
    ...     generations = 30,
    ...     population_size = 10,
    ...     mutation_rate = 0.2,
    ...     penalty = 0.02,
    ...     random_state = 42
    ... )
    >>> print(result['best_subset'])
    >>> print(result['best_score'])
    >>> print(result['selected_features'])
    """
    
    npy.random.seed(random_state)
    num_features = X.shape[1]

    # Initialize the population
    if population is None:
        population = [npy.random.choice([0, 1], size = num_features).tolist() for _ in range(population_size)]
        print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Initialized random population.")
    else:
        # Validate that the given population is of correct shape
        if len(population) != population_size or any(len(ind) != num_features for ind in population):
            raise ValueError(f"Provided population must be of shape ({population_size}, {num_features})")
        print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Using provided initial population.")

    # Store evolution and logs
    evolution = []
    logs = []

    # Global best variables
    global_best_individual = None
    global_best_score = float('-inf')

    # Genetic algorithm loop
    number_of_generations = tqdm(range(generations), desc = "Running GAFS")
    for generation in number_of_generations:

        # Print current time and generation
        # current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        # print(f"{current_time} - Starting Generation {generation + 1}/{generations}")

        # Evaluate population
        scores = []
        generation_logs = []

        for individual in population:
            score = evaluate(individual, X, y, model, metric, penalty, num_features)
            scores.append(score)

            # Log detailed information
            generation_logs.append({
                'Score': score,
                'Selected Features': sum(individual),
                'Feature Subset': individual
            })

        # Get the best score and individual of the generation
        generation_best_index = npy.argmax(scores)
        generation_best_individual = population[generation_best_index]
        generation_best_score = scores[generation_best_index]

        # Update global best
        if generation_best_score > global_best_score:
            global_best_individual = generation_best_individual
            global_best_score = generation_best_score

        # Log generation results
        logs.append({
            'Generation': generation,
            'Best Score': generation_best_score,
            'Best Individual': generation_best_individual,
            'All Evaluations': generation_logs
        })

        # Store evolution data
        evolution.append({'Generation': generation, 'Best Score': generation_best_score})

        # Create new population
        selected = roulette_selection(population, scores)  # Use roulette_selection
        new_population = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % len(selected)]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])
        population = [mutation(ind, mutation_rate, mutation_preference) for ind in new_population]

    # Prepare final results
    results = {
        'best_subset': global_best_individual,
        'best_score': global_best_score,
        'selected_features': npy.array(global_best_individual).sum(),
        'evolution': pd.DataFrame(evolution),
        'logs': logs,
        'parameters': {  # Save the function parameters
            'X': X,  # Save the original data
            'y': y,  # Save the original data
            'model': str(model),  # Save the model name as a string
            'metric': metric,
            'generations': generations,
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'mutation_preference': mutation_preference,
            'penalty': penalty,
            'random_state': random_state,
            'save_results': save_results
        }
    }

    # Save results if requested
    if save_results:
        # Create the "results/" folder if it does not exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)  # Create the directory if it does not exist

        # Generate a file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(results_dir, f"{timestamp}.joblib")  # Save in "results/"

        # Save results
        joblib.dump(results, filename)
        print(f"Results and parameters saved to {filename}")

    return results

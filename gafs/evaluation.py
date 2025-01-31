import numpy as npy
from sklearn.model_selection import cross_val_score

def evaluate(individual, X, y, model, metric, penalty, num_features):
    """
    EVALUATE THE FITNESS OF AN INDIVIDUAL BY CALCULATING ITS SCORE

    Parameters
    ----------
    individual : list of int
        A binary list representing the feature subset.
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target vector.
    model : object
        A scikit-learn-compatible classifier.
    metric : str
        Evaluation metric for cross-validation.
    penalty : float
        Penalty factor for the number of features selected.
    num_features : int
        Total number of features in the dataset.

    Returns
    -------
    float
        The fitness score of the individual, adjusted with a penalty for the number of features selected.
    """

    # Filter selected features
    X_selected = X[:, npy.array(individual, dtype = bool)]

    if X_selected.shape[1] == 0:  # Avoid empty feature subsets
        return 0.0
        
    # Evaluate using cross-validation
    score = npy.mean(cross_val_score(model, X_selected, y, cv = 5, scoring = metric))
    # Apply penalty for the number of features
    penalty_factor = sum(individual) / num_features

    return score - penalty * penalty_factor

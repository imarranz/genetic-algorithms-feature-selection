import matplotlib.pyplot as plt
import seaborn as sns
import numpy as npy

def plot_gafs_results(results, plot_type = 'evolution', ax = None, gridsize = 30, cmap = 'Blues'):
    """
    GENERATE VISUALIZATIONS FOR GENETIC ALGORITHMS FEATURE SELECTION (GAFS) RESULTS

    Parameters
    ----------
    results : dict
        Dictionary containing the results of the GAFS process.
        Keys expected: 'evolution', 'logs', 'best_subset'.
    plot_type : str
        Type of plot to generate. Options:
        - 'evolution': Evolution of the best score across generations.
        - 'distribution': Distribution of selected features across evaluations.
        - 'scatter': Scatter plot showcasing the relationship between the score achieved and a particular feature or aspect of the evaluations.
        - 'relation': Relationship between number of selected features and score (hexbin plot).
        - 'best_subset': Bar plot showing the best feature subset.
        - 'variable_frequency': Frequency of each variable appearing in the best solutions.
    ax : matplotlib.axes._axes.Axes, optional
        Matplotlib axis to plot on. If None, creates a new figure and axis.
    gridsize : int, optional
        Number of hexagons in the hexbin grid (only for 'relation'). Default is 30.
    cmap : str, optional
        Colormap for the hexbin plot (only for 'relation'). Default is 'Blues'.

    Returns
    -------
    matplotlib.axes._axes.Axes
        The axis containing the plot.

    Raises
    ------
    ValueError
        If an invalid plot_type is provided.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if plot_type == 'evolution':
        evolution = results['evolution']
        ax.plot(evolution['Generation'], npy.sort(evolution['Best Score']), marker = 'o')
        ax.set_title('Evolution of Best Score Across Generations')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Score')
        ax.grid()

    elif plot_type == 'distribution':
        selected_features = [
            log['Selected Features']
            for gen_log in results['logs']
            for log in gen_log['All Evaluations']
        ]
        sns.histplot(selected_features, kde = True, bins = 10, ax = ax)
        ax.set_title('Distribution of Selected Features')
        ax.set_xlabel('Number of Selected Features')
        ax.set_ylabel('Frequency')
        ax.grid()

    elif plot_type == 'scatter':
        scores = [
            log['Score']
            for gen_log in results['logs']
            for log in gen_log['All Evaluations']
        ]
        selected_features = [
            log['Selected Features']
            for gen_log in results['logs']
            for log in gen_log['All Evaluations']
        ]
        ax.scatter(selected_features, scores, alpha = 0.7)
        ax.set_title('Score vs. Number of Selected Features')
        ax.set_xlabel('Number of Selected Features')
        ax.set_ylabel('Score')
        ax.grid()

    elif plot_type == 'relation':
        scores = [
            log['Score']
            for gen_log in results['logs']
            for log in gen_log['All Evaluations']
        ]
        selected_features = [
            log['Selected Features']
            for gen_log in results['logs']
            for log in gen_log['All Evaluations']
        ]
        hb = ax.hexbin(selected_features, scores, gridsize = gridsize, cmap = cmap, mincnt = 1)
        ax.set_title('Score vs. Number of Selected Features (Hexbin)')
        ax.set_xlabel('Number of Selected Features')
        ax.set_ylabel('Score')
        cb = plt.colorbar(hb, ax = ax)
        cb.set_label('Frequency')

    elif plot_type == 'best_subset':
        best_subset = results['best_subset']
        ax.bar(range(len(best_subset)), best_subset)
        ax.set_title('Best Feature Subset')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Selected (1) or Not (0)')
        ax.set_xticks(range(len(best_subset)))
        ax.grid()

    elif plot_type == 'variable_frequency':
        # Calculate frequency of each variable in the best solutions
        best_solutions = [
            gen['Best Individual']
            for gen in results['logs']
        ]
        frequency = npy.sum(best_solutions, axis = 0) / len(best_solutions)

        # Plot the frequency
        ax.bar(range(len(frequency)), frequency)
        ax.set_title('Frequency of Variables in Best Solutions')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Relative Frequency')
        ax.set_xticks(range(len(frequency)))
        ax.grid()

    else:
        raise ValueError("Invalid plot_type. Choose from 'evolution', 'distribution', 'scatter', 'relation', 'best_subset', 'variable_frequency'.")

    return ax

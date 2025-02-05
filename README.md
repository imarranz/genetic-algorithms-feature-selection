# Genetic Algorithms Feature Selection (GAFS)

<p align="center">
  <img src="https://repository-images.githubusercontent.com/925117622/be278d6a-fc2d-4bf4-90e4-f97171de2ad3" alt="Genetic Algorithms Feature Selection">
</p>

<p align="center">
  <!-- Estado y contribución -->
  <a href="#">
    <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
  </a>
  <a href="LICENSE.md">
    <img src="https://img.shields.io/badge/License-MIT-red.svg?longCache=true" alt="MIT License">
  </a>
  <a href="https://github.com/imarranz/genetic-algorithms-feature-selection/pulls">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?longCache=true" alt="Pull Requests">
  </a>
</p>

<p align="center">
  <!-- Información del repositorio -->
  <a href="https://github.com/imarranz/genetic-algorithms-feature-selection">
    <img src="https://img.shields.io/github/stars/imarranz/genetic-algorithms-feature-selection?style=social" alt="GitHub Repo stars">
  </a>
  <a href="https://github.com/imarranz/genetic-algorithms-feature-selection/fork">
    <img src="https://img.shields.io/github/forks/imarranz/genetic-algorithms-feature-selection?style=social" alt="GitHub forks">
  </a>
  <a href="https://github.com/imarranz/genetic-algorithms-feature-selection/commits/main">
    <img src="https://img.shields.io/github/last-commit/imarranz/genetic-algorithms-feature-selection" alt="Last Commit">
  </a>
  <a href="#">
    <img src="https://img.shields.io/github/repo-size/imarranz/genetic-algorithms-feature-selection" alt="Repo Size">
  </a>
</p>

<p align="center">
  <!-- Información técnica -->
  <a href="#">
    <img src="https://img.shields.io/badge/Built%20With-Python-blue" alt="Python">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Config-YAML-yellow" alt="Formato YAML">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Dependencies-Scikit--learn%20%7C%20NumPy%20%7C%20Pandas%20%7C%20Matplotlib%20%7C%20Seaborn-blue" alt="Dependencies">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Notebooks%20Support-Yes-brightgreen" alt="Jupyter Notebooks Support">
  </a>
</p>

<p align="center">
  <!-- Categoría y tipo de proyecto -->
  <a href="#">
    <img src="https://img.shields.io/badge/Category-Feature%20Selection-blue" alt="Feature Selection">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Machine%20Learning-Genetic%20Algorithms-purple" alt="Machine Learning">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Maturity-Stable-green" alt="Maturity">
  </a>
</p>

## Overview

**Genetic Algorithms Feature Selection (GAFS)** is a powerful Python-based tool meticulously crafted to conduct feature selection leveraging the robust capabilities of Genetic Algorithms (GAs). Feature selection plays a pivotal role in machine learning by identifying and selecting the most influential features for predictive modeling, thereby enriching model performance while simultaneously alleviating the burden of complexity.

This versatile package offers a flexible and efficient methodology for users seeking to unravel the intricate landscape of feature importance. By harnessing the genetic principles embedded within the algorithm, **GAFS** empowers users to sift through a plethora of features, honing in on the ones that truly drive predictive accuracy and model efficiency.

Moreover, **GAFS** seamlessly integrates with a myriad of scikit-learn-compatible models and evaluation metrics, affording users the freedom to customize the optimization process according to their unique requirements and preferences. This adaptability ensures that users can tailor their feature selection strategy to suit the nuances of their datasets and modeling objectives.

In addition to its robust selection capabilities, **GAFS** offers comprehensive logging functionalities and visualizations that illuminate the evolutionary journey of feature selection. These insights not only foster a deeper understanding of the feature selection process but also provide actionable information that can guide users in making informed decisions about their models.

In essence, **GAFS** represents a cutting-edge solution for feature selection, amalgamating the elegance of Genetic Algorithms with the practical demands of modern data science. By utilizing **GAFS**, users can embark on a transformative journey towards optimizing their predictive models, unearthing key insights, and unlocking the full potential hidden within their datasets.

---

## Key Features

Within the realm of feature selection, **GAFS** (Genetic Algorithms Feature Selection) stands out as a versatile and robust tool, offering a plethora of features designed to empower users in their data-driven endeavors. Let's delve into the multifaceted capabilities that set **GAFS** apart:

### Customizable Models

- **Diversified Model Compatibility**: **GAFS** seamlessly integrates with a wide array of scikit-learn-compatible estimators, ranging from Random Forest to Logistic Regression, providing users with the flexibility to tailor their feature selection process to the specific nuances of their datasets.

### Evaluation Metrics Support

- **Diverse Metric Options**: With support for various evaluation metrics such as accuracy, *AUC*, *F1-score*, *precision*, *accuracy* and *recall*, **GAFS** equips users with the tools to gauge the performance of their models comprehensively.

### Hyperparameter Flexibility

- **Fine-Tuned Optimization**: Users have the freedom to fine-tune various hyperparameters within the genetic algorithm, including population size, number of generations, mutation rate, and penalization for the number of selected features, ensuring a carefully curated feature selection process.

### Detailed Logging

- **Insightful Tracking**: **GAFS** offers detailed logs that meticulously document the evolution of the evaluation metric, the number of selected features, and the subsets considered across generations. This granular level of tracking provides users with invaluable insights into the progression of the feature selection process.

### Outputs Overview

- **Optimal Feature Subset Identification**: Discover the best feature subset tailored to enhance model performance.

- **Evaluation Performance Insights**: Gain a comprehensive understanding of the associated evaluation performance metrics, allowing for informed decision-making.

- **Holistic Logs and Evolution Data**: Access detailed logs and evolution data that shed light on the evolutionary trajectory of feature selection, facilitating a deeper comprehension of the optimization process.

With its rich set of features and comprehensive functionalities, **GAFS** emerges as a pivotal ally for data scientists and machine learning enthusiasts seeking to navigate the intricate landscape of feature selection with precision and finesse. Elevate your feature selection experience with **GAFS** and unlock the true potential of your predictive modeling endeavors.

---

## Getting Started

This section guides you through the initial setup and basic usage of the GAFS package, ensuring you're ready to start enhancing your projects with genetic algorithms for feature selection.

### Prerequisites

Before installation, ensure that you have Python installed on your system (Python 3.6 or later is recommended). You will also need `pip` for installing packages.

### Installation From the Source

For developers or users interested in the latest features or contributing to the development of the GAFS package, installing from the source is a good option. Follow these steps:

#### Clone the repository

```
git clone https://github.com/imarranz/genetic-algorithms-feature-selection.git
```

---

## Functions

### Function `genetic_algorithms_feature_selection`

The main function to perform feature selection using Genetic Algorithms (GAs).

#### Arguments

| **Argument**       | **Type**            | **Description**                                                                                     | **Possible Values**                                                                                      | **Default**       |
|---------------------|---------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|-------------------|
| `X`                | `ndarray`          | Feature matrix containing the data.                                                                | Any dataset as a NumPy array of shape `(n_samples, n_features)`.                                         | —                 |
| `y`                | `ndarray`          | Target vector containing class labels.                                                             | Any binary or multiclass target array.                                                                   | —                 |
| `model`            | `object`           | A scikit-learn-compatible classifier.                                                              | `RandomForestClassifier`, `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`. | —                 |
| `metric`           | `str`              | Metric used for model evaluation during cross-validation.                                           | `accuracy`, `roc_auc`, `f1`, `precision`, `recall`, `log_loss`.                                           | `'accuracy'`      |
| `generations`      | `int`              | Number of generations for the genetic algorithm.                                                   | Any positive integer.                                                                                     | `50`              |
| `population_size`  | `int`              | Number of individuals in the population.                                                           | Any positive integer.                                                                                     | `20`              |
| `mutation_rate`    | `float`            | Probability of mutation for each gene in an individual.                                             | Float in the range `[0, 1]`.                                                                              | `0.1`             |
| `penalty`          | `float`            | Penalty factor for the number of selected features. Higher values penalize larger subsets.          | Any non-negative float.                                                                                   | `0.01`            |
| `random_state`     | `int` or `None`    | Seed for reproducibility.                                                                           | Any integer or `None`.                                                                                    | `None`            |

#### Returns

The function returns a dictionary with the following keys:

- `best_subset`: Binary representation of the best feature subset.
- `best_score`: Best evaluation score achieved.
- `selected_features`: Number of selected features in the best subset.
- `evolution`: DataFrame showing the best score at each generation.
- `logs`: List of detailed logs for each evaluation.

---

### Function `evaluate`

Evaluates an individual (subset of features) based on the provided model, metric, and penalization for the number of selected features.

#### Arguments

| **Argument**       | **Type**    | **Description**                                                                                      | **Example**                                                                                              |
|---------------------|------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `individual`       | `list`     | Binary list representing the selected features.                                                     | `[1, 0, 1, 1, 0]`                                                                                        |
| `X`                | `ndarray`  | Feature matrix containing the data.                                                                 | A NumPy array of shape `(n_samples, n_features)`.                                                        |
| `y`                | `ndarray`  | Target vector containing class labels.                                                              | A binary or multiclass target array.                                                                     |
| `model`            | `object`   | A scikit-learn-compatible classifier or regressor.                                                  | `RandomForestClassifier()`, `LogisticRegression()`, `XGBoostClassifier()`                                 |
| `metric`           | `str`      | Metric used for evaluation. Supports classification and regression metrics.                        | `'accuracy'`, `'roc_auc'`, `'f1_score'`, `'mse'`                                                          |
| `penalty`          | `float`    | Penalty factor for the number of selected features to balance model complexity and performance.    | `0.01`                                                                                                   |
| `num_features`     | `int`      | Total number of features in the dataset.                                                            | `10`                                                                                                     |
| `favor_selection`  | `str`      | Defines the selection preference: `"less"` favors fewer features, `"more"` promotes more selection, `"neutral"` applies standard selection. | `"less"`, `"more"`, `"neutral"` |

#### Returns

- **`float`**: The fitness score of the individual, adjusted with a penalty for the number of selected features.

---

### Function `roulette_selection`

Implements a roulette wheel selection mechanism to choose individuals for the next generation based on their scores.

#### Arguments

| **Argument**    | **Type**               | **Description**                                                                                     | **Example**                                                                                              |
|---------------------|---------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `population`         | `list`             | List of binary lists representing the population.                                                  | `[[1, 0, 1], [0, 1, 0], ...]`                                                                           |
| `scores`                | `list`             | List of scores corresponding to each individual in the population.                   | `[0.85, 0.72, ...]`                                                                                      |

#### Returns

- **`list`**: A list of individuals selected for the next generation.

---

### Function `crossover`

Performs single-point crossover between two parent individuals to produce two offspring.

#### Arguments

| **Argument**       | **Type**            | **Description**                                                                                     | **Example**                                                                                              |
|---------------------|---------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `parent1`          | `list`             | Binary list representing the first parent.                                                         | `[1, 0, 1, 0, 0]`                                                                                       |
| `parent2`          | `list`             | Binary list representing the second parent.                                                        | `[0, 1, 0, 1, 1]`                                                                                       |

#### Returns

- **`tuple`**: Two offspring as binary lists, e.g., `([1, 0, 0, 1, 1], [0, 1, 1, 0, 0])`.

---

### Function `mutation`

Applies mutation to an individual with a specified probability, flipping selected features.

#### Arguments

| **Argument**       | **Type**  | **Description**                                                                                     | **Example**                              |
|---------------------|----------|-----------------------------------------------------------------------------------------------------|------------------------------------------|
| `individual`             | `list`     | Binary list representing the individual to mutate.                                     | `[1, 0, 1, 0, 1]`                       |
| `mutation_rate`   | `float`  | Probability of flipping each gene in the individual.                                     | `0.1`                                   |
| `favor`                   | `str`     | Defines the mutation bias: `"less"` reduces selected features (`1 → 0`), `"more"` increases (`0 → 1`), `"neutral"` applies standard mutation. | `"less"`, `"more"`, `"neutral"` |

#### Returns

- **`list`**: The mutated individual, e.g., `[1, 1, 1, 0, 1]`.


## Additional Resources

For further reading and learning about genetic algorithms and evolutionary computing, the following resources are recommended:

- **Introduction to Evolutionary Computing** *by A.E. Eiben and J.E. Smith*: This book provides a comprehensive introduction to the core concepts of evolutionary computing.
- **Genetic Algorithms in Search, Optimization, and Machine Learning** *by David E. Goldberg*: One of the seminal texts in the field, perfect for understanding the fundamentals and applications of genetic algorithms.
- **Handbook of Genetic Algorithms** *by Lawrence Davis*: Offers extensive coverage on the theory and practice of genetic algorithms.
- **Essentials of Metaheuristics** *by Sean Luke*: Available online for free, this book covers a broad array of topics in metaheuristics, including genetic algorithms. [Essentials of Metaheuristics](https://cs.gmu.edu/~sean/book/metaheuristics/).
- **Complex Adaptive Systems: An Introduction to Computational Models of Social Life** *by John H. Miller and Scott E. Page*: Provides insight into the application of complex adaptive systems, including evolutionary algorithms, to social sciences.
- **Wikipedia**: This platform has extensive articles on genetic algorithms and evolutionary strategies that are maintained by experts in the field.

These resources can significantly enhance your understanding and application of genetic algorithms beyond the scope of this manual.

## Contributing

Contributions to enhance the coverage of genetic algorithms and related techniques are welcome. Please follow these steps to contribute:

- 📫 **Open an Issue**: If you have ideas or see a need for improvements, start by opening an issue in this GitHub repository.
- 🍴 **Fork and Edit**: Fork this repository, make your changes, and then submit a pull request with your contributions.
- 🔍 **Review**: Your submission will be reviewed and, if appropriate, merged into the main project.

## License

This project is licensed under the MIT License.

## Contact

For more information, suggestions, or questions, please feel free to reach out via

<p align="center">
<a href="https://www.linkedin.com/in/ibon-mart%C3%ADnez-arranz/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="LinkedIn"></a>&nbsp;&nbsp;
<a href="https://github.com/imarranz/"><img src="https://img.shields.io/badge/github-FFFFFF.svg?&style=for-the-badge&logo=Github&logoColor=black" alt="GitHub"></a>&nbsp;&nbsp;
<a href="https://x.com/imarranz/"><img src="https://img.shields.io/badge/X.com-000000?style=for-the-badge&logo=X&logoColor=white" alt="X"></a>
</p>

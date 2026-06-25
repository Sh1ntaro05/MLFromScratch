# Machine Learning from Scratch

A collection of foundational machine learning algorithms implemented entirely from scratch in Python.

## Architectural Vision

This repository is built strictly for **educational purposes and radical transparency**. The primary goal is to bridge the gap between abstract mathematical theory and code execution. Rather than relying on black-box industry frameworks, the algorithms in this repository are implemented using raw NumPy matrix operations. 

This architecture allows students, developers, and researchers to read the source code and directly observe how the underlying calculus, linear algebra, and optimization mechanisms execute under the hood. 

## Repository Structure

The directory is flatly categorized by learning paradigms to optimize scannability and avoid convoluted, multi-layered nested folder paths:

```text
MLFromScratch/
├── supervised_learning/
│   ├── decision_tree_classifier.py  
│   ├── linear_regression.py  
│   ├── logistic_regression.py  
│   ├── QDA_classifier.py
│   └── random_forest_classifier.py         
│
├── unsupervised_learning/        
│
├── reinforcement_learning/       
│
├── deep_learning/
│   ├── optimizers.py 
│   ├── layers.py 
│   ├── MLP_classifier.py  
│   └── MLP_regression.py 
│
├── utilities/ 
│   ├── cross_entropy_error.py 
│   ├── data_split.py 
│   └── softmax.py 
│
└── assets/
    └── iris_dataset/
```

## Roadmap
The framework is actively expanding. Upcoming implementations include:
* **Unsupervised Learning:** K-Means Clustering, Principal Component Analysis (PCA)
* **Reinforcement Learning:** Policy and Value Iteration (MDPs), Q-Learning, Deep RL architectures

## Installation & Usage
This project utilizes `uv` for lightning-fast Python environment management, and relies on a minimal tech stack (`numpy`, `pandas`, `scikit-learn` for data fetching).

1. Clone the repository and navigate to the root directory.
2. Ensure you have `uv` installed on your system.
3. Execute scripts as modules from the project root. For example, to run the Multilayer Perceptron classifier:
```bash
uv run -m deep_learning.MLP_classifier
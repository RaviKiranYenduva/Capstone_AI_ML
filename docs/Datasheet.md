
BBO Capstone Project – Datasheet
Motivation

This dataset was created as part of the Black-Box Optimisation (BBO) capstone project to study how optimisation strategies evolve under sparse feedback and limited evaluation budgets. The dataset supports experimentation with Bayesian Optimisation, reinforcement learning-inspired exploration, Gaussian Process surrogate modelling, neural-network approximation, and PCA-guided search refinement.

The main goal was to improve query selection for eight unknown black-box functions while balancing exploration and exploitation across multiple optimisation rounds.

Dataset Composition

The dataset contains:

Query coordinates submitted across all optimisation rounds
Corresponding function evaluation outputs
Candidate query generations
Intermediate optimisation logs
PCA-transformed search directions
Surrogate model predictions and uncertainty estimates
Data Characteristics
Numerical multidimensional input vectors
Eight separate optimisation functions
Sequential time-based query history
Sparse observations due to limited evaluation budgets

The dataset became increasingly concentrated around high-performing regions in later rounds, meaning some interior regions of the search space remain underexplored.

Collection Process

The dataset was generated iteratively throughout the capstone project.

Early Rounds
Random exploration
Broad candidate generation
High uncertainty-driven search
Mid Rounds
Gaussian Process surrogate modelling
Neural-network approximation
Standardisation and preprocessing
Later Rounds
PCA-guided directional refinement
RL-inspired epsilon-greedy exploration
Local exploitation around strong-performing regions

Each optimisation round used previous function evaluations as feedback to generate improved query candidates.

Preprocessing and Transformations

Several preprocessing steps were applied:

Recursive parsing of structured input/output files
Flattening nested arrays
Padding variable-length inputs
Standardisation using StandardScaler
PCA dimensionality reduction
Residual modelling using Gaussian Processes

The preprocessing pipeline ensured all query vectors were converted into consistent numerical formats before optimisation.

Intended Uses

The dataset is intended for:

Black-box optimisation research
Bayesian optimisation experimentation
Reinforcement learning exploration studies
Hyperparameter tuning simulations
Dimensionality reduction analysis
Inappropriate Uses

The dataset should not be used for:

Traditional supervised learning benchmarks
Real-world predictive deployment
Fairness-sensitive applications
Large-scale statistical inference

The dataset is highly specialised and sparse, making generalisation unreliable outside the capstone context.

Distribution and Maintenance

The dataset is stored within my GitHub repository and maintained as part of the BBO capstone documentation workflow.

Repository contents include:

Query history
Optimisation notebooks
Reflection reports
Model card
Weekly optimisation logs

The repository is publicly accessible for educational and reproducibility purposes.

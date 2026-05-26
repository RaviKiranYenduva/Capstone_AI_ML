
BBO Capstone Project – Model Card
Model Overview
Model Name

RL-Inspired Hybrid Bayesian Optimisation Framework

Version

Final Capstone Version

Model Type

Hybrid optimisation framework combining:

Neural Networks
Gaussian Processes
PCA-guided refinement
Reinforcement Learning-inspired exploration
Intended Use

This optimisation framework is designed for:

Black-box optimisation tasks
Sequential query optimisation
Hyperparameter tuning experiments
Exploration–exploitation analysis
Educational optimisation research
Not Intended For
Production ML deployment
Real-time decision systems
High-confidence scientific optimisation
Large-scale industrial optimisation
Model Details
Core Optimisation Strategy

The optimisation framework evolved significantly across the project.

Early Stage
Broad random exploration
Uniform candidate sampling
Basic uncertainty-driven decisions
Middle Stage
Gaussian Process surrogate modelling
Neural-network function approximation
Residual uncertainty estimation
Final Stage
PCA-guided directional refinement
RL-inspired epsilon-greedy exploration
Local exploitation around promising regions

The final framework generated thousands of candidate points per iteration and selected queries based on surrogate predictions and exploration probability.

How the Model Makes Decisions

The optimisation workflow follows these steps:

Standardise historical query data
Train neural-network surrogate models
Fit Gaussian Processes on residuals
Apply PCA to identify dominant search directions
Generate thousands of candidate query points
Predict rewards and uncertainty
Select candidates using epsilon-greedy exploration

The epsilon-greedy strategy balances:

Exploitation of strong-performing regions
Continued exploration of uncertain regions

This behaviour resembles reinforcement learning policy adaptation through iterative reward feedback.

Performance Summary

Performance was evaluated based on:

Improvement in function outputs
Stability of optimisation behaviour
Ability to refine local optima
Exploration efficiency
Strongest Behaviours Observed
PCA-guided refinement improved directional search consistency
Local exploitation improved convergence in later rounds
RL-inspired exploration reduced premature convergence risk
Surrogate modelling improved query quality over time

The framework performed best when balancing structured local refinement with limited stochastic exploration.

Assumptions

The optimisation process assumes:

Historical observations contain meaningful optimisation structure
Local regions near strong outputs remain informative
PCA directions represent dominant variation patterns
Surrogate models approximate unknown functions sufficiently well

The framework also assumes that exploration noise helps avoid local optima.

Limitations and Failure Modes
Sparse Data Limitation

Very limited observations reduce surrogate reliability and may cause unstable predictions.

Overfitting Risk

Neural networks and Gaussian Processes may overfit local regions because of small sample sizes.

Search-Space Bias

Later optimisation rounds became increasingly biased toward previously successful regions.

PCA Instability

PCA components can become unreliable when derived from extremely small datasets.

Random Exploration Risk

Epsilon-greedy exploration occasionally selects suboptimal candidates intentionally.

Ethical Considerations

Transparency and reproducibility are important aspects of this optimisation framework.

The project documents:

Query history
Optimisation assumptions
Candidate-generation logic
Model evolution
Exploration decisions

This improves interpretability and allows others to understand how optimisation decisions were made.

The framework also demonstrates the importance of documenting uncertainty and sparse-data limitations in real-world ML systems.

#### Directory Structure 
#### Experiment Aims
1. Experiment 1: Show how the zero sum version achieves communication efficiency and convergence in accuracy. 
2. Experiment 2: How the localized general sum version achieves group fairness and communication efficiency. 
    1. Define the hypothesis test for the group fairness.
3. Experiment 3: Show how the SPANE algorithm gets close to the linear programming solution.
#### Main Components
1.1 Markov Chain which models the heterogeneity in the oracle model:
    1. Heterogeneity with respect to number of data points available
    2. Heterogeneity in availablity with respect to a particular group 
1. Define the constrained switching control game:
    1. Define the state space, action space, transition probabilities, the reward function
    6. Define the switching control game
2. Algorithms for solving the switching control game:
    1. Linear Programming
    2. Value Iteration
    3. SPANE
3. Federated Learning Setup
    1. Data Generation - Based on Markov Chain
    2. Model - Client
    3. Aggregation - Server
    4. Evaluation - Server
    5. Multiple Central Learners (Servers)

#### Data
1. Define Group Fairness Task on MNIST Dataset
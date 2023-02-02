# FedMech.jl
A julia framework for personalized federated learning with mechanism models.

## Cifar10 

Steps to run the experiments
- Open Julia
- `include("examples/Cifar10.jl")`
- Then call the main function, e.g., `main(0.3, 0.01, true, true)`. The 4 arguments are respectively lambda, ratio of data to use, whether use knowledge model and federeted learning training algorithm.

## Regression 

Steps to run the experiments
- Download YearPredictionMSD dataset and put to data/YearPredictionMSD 
- The other steps are similar to above
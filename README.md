# FedMech.jl

A julia framework for personalized federated learning with mechanism models. In `examples/Cifar10.jl`, `examples/Covtype.jl` and so on, a main function is implemented. The main function takes 4 arguments: \lambda (trust level of P-KM), p (ratio of data to use), withMech (whether to use knowledge model) and withFed (which federeted learning training algorithm) , e.g., `main(0.3, 0.01, true, 1)`. The possible combination includes: 

| withMech | withFed | Method name  |
| -------- | -------| ------- |
| false    | 0      | ML      |
| true     | 0      | MLwKM   |
| false    | 1      | FL      |
| true     | 1      | FLwKM   |
| false    | 2      | ADAP    |
| true     | 2      | ADAPwKM |
| false    | 3      | DITTO   |
| true     | 3      | DITTOwKM  | 

## Cifar10 

Steps to run the experiments
- Open Julia
- `include("examples/Cifar10.jl")`
- Then call the main function, e.g., `main(0.3, 0.01, true, 1)`. 

## Covtype/Sensorless/Sector 

Steps to run the experiments
- Download correspoding dataset to data/ folder. 
- When calling the main function, add additional argument `dataPath`, e.g., `main(0.3, 0.01, true, 1; dataPath="data/covtype")`

## Regression 

Steps to run the experiments
- Download YearPredictionMSD dataset and put to data/YearPredictionMSD 
- The other steps are similar to above

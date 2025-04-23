# Constraint Satisfaction in Top-N Recommendation System

**Important** before running the ILP solver you must install valid *gurobi* license on your computer.
See https://www.gurobi.com/downloads/ for more info.

## Set up python environment

```bash
conda env create -f environment.yml
```

## Running the experiments
Some experiments are runnable on data generated on the fly and should be ready to run.
In order to run experiments that involve the recommendation model and data, use the Movielens 20M dataset:

* Download the dataset from https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
* Extract the files it in the folder *data/movielens_raw*
* Run scripts *data_transform* and *data_aggregator*


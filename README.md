# Predicting Data Correlation via Large Language Models

This project aims at predicting which column pairs are likely correlated by analyzing column names via large language models (LLMs). The following instructions have been tested on a p3.2xlarge AWS EC 2 instance with Deep Learning Base GPU AMI (Ubuntu 20.04) 20230804.

## Setup

1. Download this repository, e.g., run:
```
git clone https://github.com/itrummer/DataCorrelationPredictionWithNLP
```
2. Install required packages:
```
cd DataCorrelationPredictionWithNLP
pip install -r requirements.txt
```
3. Download correlation data [here](https://drive.google.com/file/d/14w73DDKaCaNpaFM0CwR-ES-wkcsIYX99/view?usp=share_link), e.g., run (from within the `DataCorrelationPredictionWithNLP` directory):
```
wget -O correlationdata.csv "https://docs.google.com/uc?export=download&confirm=t&id=14w73DDKaCaNpaFM0CwR-ES-wkcsIYX99"
```

## Running Experiments

The majority of experiments in the associated paper are realized via the code at
```
src/dp/experiments/run_experiment.py
```
This script takes the following command line parameters:
- Path to the .csv file (downloaded before) with correlation data.
- The correlation coefficient to calculate (e.g., "pearson").
- The minimal coefficient value for a correlation (e.g., 0.9).
- The maximal p-value for a correlation (e.g., 0.05 - set to 1 for Theil's U).
- The model type used for prediction (e.g., "roberta").
- The specific model used for prediction (e.g., "roberta-base").
- Whether training and test samples derive from common data ("defsep" vs. "datasep").
- The ratio of samples used for testing (e.g., 0.2), rather than training.
- Whether to use column types for prediction (1) or not (0).
- Output path to file with experimental results (will be created).

Upon invocation, the script trains a language model for correlation prediction, then uses it on the test set to calculate various metrics of prediction performance. It writes the resulting metrics for this approach, as well as for a simplistic baseline, to the output file.

E.g., try running (from within the `DataCorrelationPredictionWithNLP` directory, assuming that python3.9 is the Python interpreter):
```
PYTHONPATH=src python3.9 src/dp/experiments/run_experiment.py correlationdata.csv pearson 0.9 0.05 roberta roberta-base defsep 0.2 0 predictionresults.csv
```
Note: the number of epochs was reduced to five in the latest version which does not appear to reduce prediction quality. Set `num_train_epochs` in `run_experiment.py` to 10 to reproduce paper experiments.

## Analyzing Results

If executing the code above, results will be stored in `predictionresults.csv`. This CSV file contains results for the LLM predictor as well as for a simple baseline. For instance, search for the row containing the string `1-final` to find LLM results for all test cases.

The semantics of the file columns is the following:
- coefficient: predicting correlation according to this correlation coefficient (e.g., pearson).
- min_v1: minimal absolute value of correlation coefficient to be considered correlated.
- max_v2: maximal p-value to be considered correlated.
- mod_type: the type of language model (e.g., roberta).
- mod_name: the name of the language model (e.g., roberta-base).
- scenario: whether column pairs in training and test set may derive from the same tables (defsep) or not (datasep).
- test_name: whether LLM predictor or baseline, possibly data subset used for testing (e.g., long column names).
- pred_method: whether result for simple baseline (0) or LLM predictor (1).
- lb: lower bound on metric defining data subset (e.g., column name length in characters).
- ub: upper bound on metric defining data subset (e.g., column name length in characters).
- f1: F1 score for predicting correlation.
- precision: precision when predicting correlated column pairs.
- recall: recall when predicting correlated column pairs.
- accuracy: accuracy when predicting correlated column pairs.
- mcc: Matthew's Correlation Coefficient when predicting correlated column pairs.
- prediction_time: time (in seconds) used for prediction.
- training_time: time (in seconds) used for training LLM predictor.

Note that the first rows report results for the simple baseline, not for the LLM predictor.

## How to Cite

This code relates to the following papers:

```
@article{Trummer2021nlp,
author = {Trummer, Immanuel},
doi = {10.14778/3450980.3450984},
issn = {21508097},
journal = {Proceedings of the VLDB Endowment},
number = {7},
pages = {1159--1165},
title = {{The case for nlp-enhanced database tuning: Towards tuning tools that “read the manual”}},
volume = {14},
year = {2021}
}
```

```
@article{Trummer2023d,
author = {Trummer, Immanuel},
journal = {PVLDB},
number = {13},
title = {{Can Large Language Models Predict Data Correlations from Column Names?}},
volume = {16},
year = {2023}
}
```

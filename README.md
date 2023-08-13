# DataCorrelationPredictionWithNLP
In this project, I try to predict data correlations by analyzing column names. Such methods can be useful, for instance, for database tuning as correlations make optimization hard. I use pre-trained language models, based on the transformer architecture, for the prediction.

The scripts used for experiments are in this repository. Furthermore, a file containing results about data correlations in thousands of Kaggle data sets is available for download [here](https://drive.google.com/file/d/14w73DDKaCaNpaFM0CwR-ES-wkcsIYX99/view?usp=share_link).

The majority of experiments in the associated paper are realized via the code at
```
src/dp/experiments/coefficients.py
```
This script takes the following command line parameters:
- The correlation coefficient to calculate (e.g., "pearson").
- The minimal coefficient value for a correlation (e.g., 0.9).
- The maximal p-value for a correlation (e.g., 0.05 - set to 1 for Theil's U).
- The model type used for prediction (e.g., "roberta").
- The specific model used for prediction (e.g., "roberta-base").
- Whether training and test samples derive from common data ("defsep" vs. "datasep").
- The ratio of samples used for testing (e.g., 0.2), rather than training.

Note that the path to the input file is currently hard-coded. It needs to be changed to point to the file referenced above.

Upon invocation, the script trains a language model for correlation prediction, then uses it on the test set to calculate various metrics of prediction performance. It outputs those metrics via the wandb online platform, as well as to a local file on hard disk. It also generates breakdowns of prediction results, evaluating how test case properties (e.g., column name length) influence prediction performance.

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
@inproceedings{Trummer2021a,
author = {Trummer, Immanuel},
booktitle = {https://arxiv.org/pdf/2107.04553.pdf},
pages = {1--12},
title = {{Can deep neural networks predict data correlations from column names?}},
year = {2021}
}
```

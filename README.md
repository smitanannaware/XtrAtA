# Extraction of Surprising Aspects from Customer Reviews: Datasets and Experiments with Language Models
This repository contains the datasets described in the paper above.

## Dataset
The datasets are available in the dataset directory for restaurant, hotel, and hair salon domains. Each dataset has a train_test and dev directory with data in Excel format, and the train_test folder contains the data used in 10-fold evaluation.

The table below shows details of the columns in the dataset.
| Column | Description |
| ------ | ----------- |
| review | The user review text for the domain. |
| true_strong | Extractive annotation for strong atypical aspects. |
| true_weak | Extractive annotation for weak atypical aspects. |
| true_other | Extractive annotation for other atypical aspects which do not fall in the strong or weak category. |
| true_strong_weak | Extractive Strong and Weak atypical aspects are combined. |
| abs_true_strong | Abstractive annotation for strong atypical aspects. |
| abs_true_weak | Abstractive annotation for strong atypical aspects. |
| abs_true_other | Abstractive annotation for other atypical aspects which do not fall in the strong or weak category. |
| abs_true_strong_weak | Abstractive Strong and Weak atypical aspects are combined. |



# Code Smell Identification: A Comparison between Metric-based and NLP-based Classification

This repository contains the code and dataset for comparing the identification of code smells using metric-based and NLP-based models. Various machine learning models were used for the comparison, including BERT, CodeBERT, LongFormer, a Multi-Layer Perceptron, and RandomForest.

## NLP Tested Models

The NLP models used for code smell identification were obtained from the Hugging Face library. The models include:

- BERT
- CodeBERT
- LongFormer

## Metric-based Classification Models

In addition to NLP models, metric-based classification models were also used. The following models were employed:

- Traditional RandomForest
- RandomForest with word2vec
- Multilayer Perceptron

## Models used for the final comparison
- Traditional RandomForest
- RandomForest with word2vec
- CodeBERT

## Evaluation Metrics

The following metrics were used to assess the model performances:

- Precision
- Recall
- Accuracy
- F1-score
- MCC (Matthews Correlation Coefficient)

## Statistical Tests

To compare the models' performances, the non-parametric Friedman test was employed.

## Dataset

The dataset used for training and evaluating the models can be found in the "dataset" folder and is named "ultimate_dataset.csv".

## Repository Structure

- `ML_models/` contains the traditional ML models used
- `NLP_models/` contains the deep learning models used
- `dataset/` contains the various datasets used
- `results/` will contain the evaluation results of the models
- `README.md` (this file) provides information on how to use the repository

## Usage Instructions

1. Ensure Python is installed (recommended version: Python 3.0)
2. Clone the repository to your local system: `git clone https://github.com/DarioDeMaio/Code_Smells_NLP`

## License

This project is licensed under the MIT License. For more information, refer to the `LICENSE` file.


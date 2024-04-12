# SNLP-project

# Natural language processing project

## Date: 12 April 2024

## Authors: Edvard Ohlström, Eemeli Astola, Linus Lind, Heikki Penttinen

## LICENSE: GNU GPLv3

## Overview

In this project we implment several models for sentiment analysis of speech data. Modles used are Bag of words, CNN, LSTM and BERT. Comparison can be found in the [text](model_comparison.ipynb) and detailed information can be found in the PDF.

Here is an overview of the project's file structure and a brief description of the main components:

```
├── PyTorch-bert.ipynb           # Jupyter notebook for the BERT model training and evaluation
├── PyTorch-cnn.ipynb            # Jupyter notebook for the CNN model training and evaluation
├── PyTorch-lstm.ipynb           # Jupyter notebook for the LSTM model training and evaluation
├── model_comparison.ipynb       # Jupyter notebook for comparing different models
├── data                         # Directory for datasets (Dataset was private, thus not included)
│ ├── dev_2024.csv                  # Development dataset
│ ├── test_2024.csv                 # Test dataset
│ └── train_2024.csv                # Training dataset
├── model                        # Directory for model weights
│   ├── cnn.pth
│   ├── lstm.pth
│   └── pytorch_bert_big_train.h5
├── probability-based-BOW        # Custom Bag of Words model implementation
│ ├── LICENSE                       # License file for the software
│ ├── README.md                     # README for the BOW model implementation
│ ├── gitignore.txt                 # Specifies intentionally untracked files to ignore
│ ├── metrics.py                    # Python script for evaluating model metrics
│ ├── model.py                      # Python script defining the BOW model
│ ├── preprocessing.py              # Python script for data preprocessing
│ ├── requirements.txt              # Required libraries and dependencies to run the BOW model
│ └── run.py                        # Main executable script for the BOW model
├── bow_test_results.csv         # CSV containing test results from the Bag of Words model
├── bow_train_results.csv        # CSV containing training results from the Bag of Words model
├── group6_bert_submission.csv   # Final Kaggle submission CSV for the BERT model predictions
├── README.md                    # Project overview and general information
└── requirements.txt             # List of dependencies for project setup
```

## Installation

This was developed with python 3.11.8

$ pip install -r requirements.txt

## Data

The data was given by the kaggle competition has been pre-partitioned into 3 sets: train,dev,test

The test dataset consists of two parts: public and private,

Observe! The text column may contain unclosed quotation ("), which causes issues for pandas. To properly load the data use pandas.read_csv("\*.csv", quoting=3)

The distribution of toxic and non-toxic texts in the training and test sets is different, and the scores achieved on the public test set are not the final results, the performance on the hidden test data will also be taken into account while we grade the solutions
Files

    train_2024.csv - the training set, 99000 rows
    dev_2024.csv - the development set, 11000 rows
    test_2024.csv - the test set, 12001 rows

Columns

    id - column for the example_id within the set
    text - the text of the comment
    label - binary label (1=Toxic/0=NonToxic

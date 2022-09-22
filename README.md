# Overview

This code is written to fulfill the requirements of my bachelor thesis

### Aims

The Objective is to fine-tune models for the ORKG completion.

### Approach

The approach is simple; I prepared ORKG data and format it to a QA with context dataset
and use it to train models already pretrained using the Squad v2 dataset

# How to Run

first prepare the dataset -> for this run the ./data_preparation/main.py
second run the ./training_evaluation/main.py to train and evaluate the models

### Prerequisites

Experience with python might be needed to understand and run the script

### Software Dependencies

All the dependencies are described in the ./requirements.txt
you can use 'pip install -r requirements.txt' to install the dependencies
The script was written using Python version 3.10 (other version might also be compatible)

Also, you need to run "python -m spacy download en_core_web_sm" to download the spacy pipeline we use in the data
preparation part of the script

### Service Retraining

Follow the ./data_preparation/README.md file and the ./training_evaluation/README.md file

### Contribution

This service is developed by:
Moussab Hrou, moussab.hrou@stud.uni-hannover.de

### References

The code in ./data_preparation/fetch_abstracts.py is
from [link](https://gitlab.com/TIBHannover/orkg/orkg-bioassays-semantification/-/blob/master/services/metadata.py)
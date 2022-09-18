# Overview

### Aims
This code is part of my bachelor thesis.
I finetune Squad v2 based models using data from the orkg to predict object labels

### Approach
The approach is simple; I prepared ORKG data and format it to a QA with context dataset
and use it to train models already pretrained using the Squad v2 dataset


# How to Run

### Prerequisites
Experience with python might be needed to understand and run the script

### Software Dependencies
All the dependencies are described in the ./requirements.txt
you can use 'pip install -r requirements.txt' to install the dependencies
The script was written using Python version 3.10 (other version might also be compatible)



### Service Retraining
Here some text about how to re-build the dataset and re-train the model.

git clone <link to your repository>
cd <repository directory>
pip install -r requirements.txt
python -m src.main [any necessary arguments]


or

git clone <link to your repository>
cd <repository directory>
pip install -r requirements.txt
python -m src.main -s dataset [any necessary arguments]
// intermediate step e.g.: run notebooks/train.ipynb and store the output model locally.
python -m src.main -s evaluate [any necessary arguments]



Service Integration
Here some text about how to use the existing model as an End-to-End service. Please consider
following the integration requirements
if you want your service to be integrated into orkgnlp.

git clone <link to your repository>
cd <repository directory>
pip install -r requirements.txt
python -m src.models.predict [any necessary arguments]



### Contribution
This service is developed by:
Moussab, Hrou asv3.771g@gmail.com



### License

### References
The code in ./data_preparation/fetch_abstracts.py is from [link](https://gitlab.com/TIBHannover/orkg/orkg-bioassays-semantification/-/blob/master/services/metadata.py)
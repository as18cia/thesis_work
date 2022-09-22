# How to run:

First the following env variable are required (otherwise could be inserted directly into the code in orkg_client.py):
* orkg_email (the login email of the ORKG service)
* orkg_password (the login passwod)
run the training-evaluation/main.py module -> at the end the final_dataset.csv file will be created in ./data/precessed
folder which will contain the un-split dataset. The split (training and evaluation) dataset variants can be found in
./data/training_ready_data folder.

# note:
the data preparation can take a long time especially fetching the abstracts can take 20+ hours.


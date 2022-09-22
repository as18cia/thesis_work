# How to run
The training script depends on the files generated from the data preparation script, precisely the files that should
exist in the ./data/training_ready_data folder.
To run the script simply run the ./training_evaluation/main.py, there you can change the vanilla models the learning rates
also there is the option to specify
    * training_test_len
    * eval_test_len
specifying these 2 parameters will make the trainer train on a small (the number specified) number of data from the dataset
this is meant to test first that everything works fine.
not specifying them will ensure the training happens on the whole dataset

# results
The best models with their evaluation results will be found in ./data/trained_models_results
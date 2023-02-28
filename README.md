# FAC-BERT: A faithful attention-based model for the classification of crisis events

1.  eval: cross-validation
    python run.py -mode eval - event_type typhoon
        event_type: typhoon/quake
2.  train
    python run.py -mode train -event_type typhoon -saved_model_path ../data/saved_models/ 
3.  prediction 
    python run.py -mode prediction -event_type quake -saved_model_path ../data/saved_models/ -input_new_data_path ../data/unlabeled_data/new_data.csv -output_new_data_path ../data/output_data/new_data.csv

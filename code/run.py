import os 
import sys 
import torch 
import argparse
import random
import numpy as np
import pandas as pd
import utils
import re
from mtlTrainer import MTLTrainer 
from clsTrainer import CLSTrainer
from trainer import FACTrainer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

random.seed(12345)
np.random.seed(67891)
torch.manual_seed(54321)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-input_path', type = str, default = '../data/labeled_data/examples.csv')
    parser.add_argument('-random_state', type = int, default = 12)
    parser.add_argument('-model_config', type = str, default = 'vinai/bertweet-base')
    parser.add_argument('-cls_hidden_size', type = int, default = 128)
    parser.add_argument('-exp_hidden_size', type = int, default = 128)
    parser.add_argument('-ignored_exp_percent', type = float, default = 0.5)
    parser.add_argument('-ignored_cls_percent', type = float, default=0)
    parser.add_argument('-att_weight', type= float, default=0.5)
    parser.add_argument('-lr', type = float, default = 2e-5, help = 'learning rate')
    parser.add_argument('-max_len', type = int, default = 128)
    parser.add_argument('-n_epochs', type = int, default = 10)
    parser.add_argument('-n_folds', type = int, default = 5)
    parser.add_argument('-patience', type = int, default = 3)
    parser.add_argument('-test_size', type = int, default = 0.15)
    parser.add_argument('-train_batch_size', type = int, default = 16)
    parser.add_argument('-test_batch_size', type = int, default = 128)
    parser.add_argument('-sep_exp_token', type = str, default = ' _sep_exp_token_ ')
    parser.add_argument('-id_col', type = str, default = 'tweet_id')
    parser.add_argument('-text_col', type = str, default = 'tweet_text')
    parser.add_argument('-label_col', type = str, default = 'corrected_label')
    parser.add_argument('-exp_col', type = str, default = 'informative_content')
    parser.add_argument('-device', type = str, default = 'cuda:1')
    parser.add_argument('-input_new_data_path', type = str, default = "../data/unlabeled_data/new_data.csv")
    parser.add_argument('-output_new_data_path', type = str, default = "../data/output_data/new_data.csv")
    parser.add_argument('-saved_model_path', type = str, default = "../data/saved_models/")
    
    parser.add_argument('-full_train', type = str, default = True, 
            help = 'True: full data, False: only use the correct prediction from 1st phase for training in 2nd phase')
    
    parser.add_argument('-mode', type = str, default = 'eval', help='train: train model, eval:evaluate with n-folds, \
                                prediction: make prediction on new data')

    parser.add_argument('-event_type', type = str, default = 'quake')


    label_event_map = {'quake': ['injured_or_dead_people', 'affected_people_and_evacuations', \
                        'infrastructure_and_utilities_damage', 'rescue_volunteering_and_donation_effort',  \
                        'other_useful_information', 'not_related_or_irrelevant'],
                        'typhoon': ['caution_and_advice', 'affected_people_and_evacuations', \
                        'infrastructure_and_utilities_damage', 'rescue_volunteering_and_donation_effort',  \
                        'other_useful_information', 'not_related_or_irrelevant']}
    prepro_exp = 'prepro_exp'
    prepro_text = 'prepro_text'
    prepro_label = 'prepro_label'
    

    args = parser.parse_args()

    labels = label_event_map[args.event_type]
    label_idx_map = {label:i for i, label in enumerate(labels)}
    idx_label_map = {idx: label for label, idx in label_idx_map.items()}
    
    if args.mode == 'prediction':
       
        trainer = FACTrainer(args)
        cls_classes = len(labels)
        print("Load models...")
        trainer.load(cls_classes = cls_classes, saved_model_path = args.saved_model_path+args.event_type+"/model.pt")
        

        print("Read new data...")
        new_data = []
        prepro_data = []
        with open(args.input_new_data_path, "r") as f:
            for line in f.readlines():
                new_data.append(line.strip())
                prepro_data.append(utils.preprocess_text(line.strip()))

        cls_pred_p1, _, exp_pred_data, exp_pred_masked, att_scores = trainer.classify(prepro_data)
       

        print("Write outputs...")
        with open(args.output_new_data_path, "w") as f:
            f.write("text\tlabel\trationale_label\n")
            for i in range(len(new_data)):
                f.write("{}\t{}\t{}\n".format(new_data[i], idx_label_map[int(cls_pred_p1[i])], \
                      att_scores[i]))
        print("Done!")

    else:
        # load data 
        data = pd.read_csv(args.input_path)

        # preprocess exp
        data[prepro_exp] = data[args.exp_col].apply(lambda x: x.strip().replace('[SEP]', args.sep_exp_token))
        data[prepro_exp] = data[prepro_exp].apply(lambda x: utils.preprocess_text(x))
        data[prepro_exp] = data[prepro_exp].apply(lambda x: [y.strip() for y in x.split(args.sep_exp_token)])
        # preprocess text
        data[prepro_text] = data[args.text_col].apply(lambda x: utils.preprocess_text(x))
        
        data[prepro_label] = data[args.label_col].apply(lambda x: label_idx_map[x])

        data.drop_duplicates(subset = prepro_text, inplace = True)
        print("Data: ", data.shape)

        # initialize models
        text_data = np.array(data[prepro_text])
        exp_data = np.array(data[prepro_exp])
        cls_data = np.array(data[prepro_label])
        

        if args.mode == 'train':
            #train models on entire data
            trainer = FACTrainer(args, text_data, cls_data, exp_data)
            trainer.train(saved_model_path = args.saved_model_path + args.event_type+"/model.pt")

        elif args.mode == "eval":
            trainer = FACTrainer(args, text_data, cls_data, exp_data)
           
            #cross-validate 
            kfold = StratifiedShuffleSplit(n_splits = args.n_folds, test_size = args.test_size, random_state = args.random_state)
            fold = 0
            cls_f1s = []
            exp_f1s = []
            att_f1s = []
            for train_indices, remain_indices in kfold.split(text_data, cls_data):
                valid_indices, test_indices = train_test_split(remain_indices, test_size = 0.5, random_state = args.random_state,
                                    stratify = cls_data[remain_indices])
                print("---------------------FOLD {}-----------------------".format(fold))
                cls_f1, exp_f1_p1, exp_f1_p2 = trainer.eval(train_indices, valid_indices, test_indices)
                cls_f1s.append(cls_f1)
                exp_f1s.append(exp_f1_p1)
                att_f1s.append(exp_f1_p2)
                print("--------------------------------------------------")
                fold+=1

            print("Final evaluation--------------------")
            print("cls_f1: {}, exp_f1: {}, att_f1: {}".format(np.mean(cls_f1s), np.mean(exp_f1s), np.mean(att_f1s)))

    




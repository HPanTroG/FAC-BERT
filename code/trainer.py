import transformers
from transformers import AutoTokenizer, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from facbert import FACBERT
import numpy as np
import torch
import torch.nn as nn
import sys
import utils
import random
import time


class FACTrainer:
    def __init__(self, args, data = None, cls_labels = None, exp_labels = None):
        self.args = args 
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_config)
        if data is not None:
            self.data = np.array([self.tokenizer.cls_token + " " + x + " " + self.tokenizer.sep_token for x in data])
            self.exp_labels = exp_labels
            self.cls_labels = cls_labels 
            self.cls_classes = len(set(cls_labels))

            # preprocess and convert data to tensor
            self.cls_labels = torch.tensor(cls_labels, dtype = torch.long)
            self.tokenized_data, self.input_ids, self.attention_masks, self.tokenized_data_slides = \
                utils.tokenize_text(self.tokenizer, self.data, self.tokenizer.pad_token)
            self.exp_labels_mapping = utils.map_exp_labels(self.tokenizer, self.data, exp_labels)
            self.tokenized_data, self.input_ids = np.array(self.tokenized_data,dtype = object), torch.tensor(self.input_ids, dtype = torch.long)
            self.attention_masks, self.tokenized_data_slides = torch.tensor(self.attention_masks, dtype = torch.long), np.array(self.tokenized_data_slides,dtype = object)
            self.exp_labels_mapping = torch.tensor(self.exp_labels_mapping, dtype = torch.long)

    def fit(self, input_ids, attention_mask, cls_labels = None, exp_labels = None, cls_criterion = None, 
            exp_criterion = None, batch_size = 64, mode = 'cls'):
        
        total_loss = 0
        n_batches = int(np.ceil(len(input_ids)/batch_size))
        epoch_indices = random.sample([i for i in range(len(input_ids))], len(input_ids))

        self.model.train()
        for batch_start in range(0, len(epoch_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(epoch_indices))
            batch_indices = epoch_indices[batch_start: batch_end]
            batch_input_ids = input_ids[batch_indices]
            batch_attention_masks = attention_mask[batch_indices]
            batch_exp_labels = exp_labels[batch_indices]

            cls_preds, attention_weights, exp_preds = self.model(batch_input_ids.to(self.args.device),
                                                        batch_attention_masks.to(self.args.device))
            
            if mode == 'exp':
                
                if self.args.ignored_exp_percent == 0:
                    loss = exp_criterion(exp_preds, batch_exp_labels.to(self.args.device).float()).mean(dim = -1).sum()
                else:
                    idx = batch_exp_labels.sum(dim = -1) >=0
                    loss = exp_criterion(exp_preds[idx, :], batch_exp_labels[idx, :].to(self.args.device).float()).mean(dim = -1).sum()
            
            elif mode == 'cls':
                batch_cls_labels = cls_labels[batch_indices]

                att_criterion = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)
                cls_loss = cls_criterion(cls_preds, batch_cls_labels.to(self.args.device)).mean(dim = -1).sum()
                att_loss = 1 - att_criterion (attention_weights, batch_exp_labels) - 0.1 # batch_exp_labels: probability

                att_loss[att_loss <= 0] = 0
                att_loss = att_loss.mean()

                loss = cls_loss +self.args.att_weight + att_loss
            
            self.optimizer.zero_grad()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

        return total_loss/n_batches

    def predict(self, input_ids, attention_masks, cls_labels=None, exp_labels=None, cls_criterion=None, exp_criterion=None, batch_size=128, mode = 'exp'):
        """make prediction"""
        self.model.eval()

        cls_preds = []
        exp_preds = []
        exp_probs = []
        cls_probs = []
        attention_outputs = None
        batch_start = 0
        total_loss = 0 
        batch_num = 0
        with torch.no_grad():
            for batch_start in range(0, len(input_ids), batch_size):
                batch_end = min(batch_start + batch_size, len(input_ids))
                batch_input_ids = input_ids[batch_start: batch_end]
                batch_attention_masks = attention_masks[batch_start: batch_end]
                
                cls_outs, attention_weights, exp_outs = self.model(batch_input_ids.to(self.args.device), 
                                batch_attention_masks.to(self.args.device))
    
                if cls_labels is not None:
                    batch_cls_labels = cls_labels[batch_start: batch_end]
                    batch_exp_labels = exp_labels[batch_start: batch_end]
                    
                    if mode == 'exp':
                        loss = exp_criterion(exp_outs, batch_exp_labels.to(self.args.device).float()).mean(dim = -1).sum()
                    else:
                        att_criterion = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)
                        cls_loss = cls_criterion(cls_outs, batch_cls_labels.to(self.args.device)).mean(dim = -1).sum()
                        att_loss = 1 - att_criterion(attention_weights, exp_outs) - 0.1
                        att_loss[att_loss <=0] = 0
                        att_loss = att_loss.mean()
                        loss = cls_loss + self.args.att_weight * att_loss
                    total_loss += loss.item()
                
                cls_outs = nn.Softmax(dim = -1)(cls_outs)
                cls_outs = cls_outs.max(dim = -1)
                cls_pred_labels = cls_outs.indices.cpu()
                cls_pred_probs = cls_outs.values.cpu()

                exp_probs += exp_outs.cpu()
                exp_preds += torch.round(exp_outs).long().cpu()  
                cls_preds += cls_pred_labels
                cls_probs += cls_pred_probs
                if attention_outputs is None:
                    attention_outputs = attention_weights.cpu()
                else:
                    attention_outputs = torch.cat((attention_outputs, attention_weights.cpu()), 0)

                batch_num += 1  

        return cls_preds, exp_preds, cls_probs, exp_probs, total_loss/batch_num, attention_outputs


    def eval(self, train_indices = None, valid_indices = None, test_indices= None):
        cls_criterion = torch.nn.CrossEntropyLoss(reduction = 'none', ignore_index = -1)
        exp_criterion = torch.nn.BCELoss(reduction = 'none')

        """train, eval, test"""
        train_data, valid_data, test_data = self.data[train_indices], self.data[valid_indices], self.data[test_indices]
        train_input_ids, valid_input_ids, test_input_ids = self.input_ids[train_indices], self.input_ids[valid_indices], self.input_ids[test_indices]
        train_attention_masks, valid_attention_masks, test_attention_masks = self.attention_masks[train_indices], self.attention_masks[valid_indices], self.attention_masks[test_indices]
        train_tokenized_data_slides, valid_tokenized_data_slides, test_tokenized_data_slides = self.tokenized_data_slides[train_indices], self.tokenized_data_slides[valid_indices], self.tokenized_data_slides[test_indices]
        train_exp_labels, valid_exp_labels, test_exp_labels = self.exp_labels_mapping[train_indices], self.exp_labels_mapping[valid_indices], self.exp_labels_mapping[test_indices]
        train_cls_labels, valid_cls_labels, test_cls_labels = self.cls_labels[train_indices], self.cls_labels[valid_indices], self.cls_labels[test_indices]
        ignored_cls_indices = None
        ignored_exp_indices = None

        # sample x% data without labels
        if self.args.ignored_exp_percent != 0:
            _, ignored_exp_indices = train_test_split([i for i in range(len(train_cls_labels))], 
                                                      test_size = self.args.ignored_exp_percent, random_state = self.args.random_state,
                                                      stratify = train_cls_labels)
            train_exp_labels[ignored_exp_indices] = -1
        if self.args.ignored_cls_percent != 0:
            _, ignored_cls_indices = train_test_split([i for i in range(len(train_cls_labels))], \
                                                      test_size = self.args.ignored_cls_percent, random_state = self.args.random_state,
                                                      stratify = train_cls_labels)
            train_cls_labels[ignored_cls_indices] = -1

        # initialize base model
        self.model = FACBERT(model_config = self.args.model_config, cls_hidden_size = self.args.cls_hidden_size, 
                    exp_hidden_size = self.args.exp_hidden_size, cls_classes = self.cls_classes)
        self.model.to(self.args.device)
        n_batches = int(np.ceil(len(train_indices)/self.args.train_batch_size))
        self.optimizer = AdamW([{'params':[ param for name, param in self.model.named_parameters() if 'cls_layers' not in name]}], lr = self.args.lr, eps = 1e-8)
        total_step = n_batches * self.args.n_epochs
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                num_warmup_steps=0,
                                                                num_training_steps=total_step)

        # phase 1 prediction
        print(">> Phase 1........................................")
        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            train_loss = self.fit(train_input_ids, train_attention_masks, train_cls_labels, train_exp_labels, 
                                  cls_criterion, exp_criterion, self.args.train_batch_size, mode = 'exp')
            #evaluate on valid set
            _, _, _, _, valid_loss, _ = self.predict(valid_input_ids, valid_attention_masks, 
                                                                                 valid_cls_labels, valid_exp_labels, cls_criterion, exp_criterion, self.args.test_batch_size, mode='exp')
            print("++++++Epoch: {}, train_loss: {}, valid_loss: {}".format(epoch, train_loss, valid_loss, time.time() - begin_time))

        #extract exp probs on whole train set 
        _, _, _, exp_pred_train_probs, _, _ = self.predict(train_input_ids, train_attention_masks, train_cls_labels,
                                                           train_exp_labels, cls_criterion, exp_criterion, self.args.test_batch_size, mode = 'exp')
        exp_pred_train_probs = torch.stack(exp_pred_train_probs, dim = 0).to(self.args.device)
        # evaluate on test set
        cls_pred_labels, exp_pred_labels_p1, _, _, _, _ = self.predict(test_input_ids, test_attention_masks,
                                                test_cls_labels, test_exp_labels, cls_criterion, exp_criterion, self.args.test_batch_size, mode='exp')
        exp_true = utils.max_pooling(test_exp_labels, test_tokenized_data_slides, test_data)
        exp_pred_p1  = utils.max_pooling(exp_pred_labels_p1, test_tokenized_data_slides, test_data)

        exp_f1_p1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred_p1)])
        exp_precision_p1 = np.mean([precision_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred_p1)])
        exp_recall_p1 = np.mean([recall_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred_p1)])
        cls_f1_p1 = f1_score(test_cls_labels, cls_pred_labels, average = 'macro')
        print('----------------------Stage 1-------------------------')
        print("cls_f1: {}, exp_f1: {}:, exp_p: {}, exp_r: {}\n".format(cls_f1_p1, exp_f1_p1, exp_precision_p1, exp_recall_p1))

        #phase 2 prediction 
        print(">> Phase 2........................................")
        self.optimizer = AdamW([{'params':[ param for name, param in self.model.named_parameters() if 'exp_layers' not in name]}], lr = self.args.lr, eps = 1e-8)
        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            train_loss = self.fit(train_input_ids, train_attention_masks, train_cls_labels, exp_pred_train_probs, 
                                  cls_criterion, exp_criterion, self.args.train_batch_size, mode = 'cls')
            #evaluate on valid set
            _, _, _, _, valid_loss, _ = self.predict(valid_input_ids, valid_attention_masks, 
                                                                                 valid_cls_labels, valid_exp_labels, cls_criterion, exp_criterion, self.args.test_batch_size, mode='cls')
            print("++++++Epoch: {}, train_loss: {}, valid_loss: {}".format(epoch, train_loss, valid_loss, time.time() - begin_time))
        
        # evaluate on test set
        cls_pred_labels, exp_pred_labels, _, exp_pred_probs,_, attentions = self.predict(test_input_ids, test_attention_masks, 
                                test_cls_labels, test_exp_labels, cls_criterion, exp_criterion, self.args.test_batch_size, mode = 'cls')
        exp_true = utils.max_pooling(test_exp_labels, test_tokenized_data_slides, test_data)
        exp_pred = utils.max_pooling(exp_pred_labels, test_tokenized_data_slides, test_data)

        topk_attentions = []
        for i in range(len(exp_pred_labels_p1)):
            k = int(sum(exp_pred_labels_p1[i]))
            if k != 0:
                # print("k: {}, attention[i]: {}".format(k, attentions[i].topk(k)))
                mink = min(attentions[i].topk(k).values)
                topk_attentions.append(list(np.array(attentions[i].numpy()>=mink.numpy())*1))
            else:
                topk_attentions.append([0 for i in range(len(attentions[i]))])

        exp_attention_pred = utils.max_pooling(topk_attentions, test_tokenized_data_slides, test_data)

        exp_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
        exp_precision = np.mean([precision_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
        exp_recall = np.mean([recall_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
        
        exp_att_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_attention_pred) if sum(y_true)!=0])
        exp_att_p = np.mean([precision_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_attention_pred) if sum(y_true)!=0])
        exp_att_r = np.mean([recall_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_attention_pred) if sum(y_true)!=0])
        cls_f1 = f1_score(test_cls_labels, cls_pred_labels, average = 'macro')

        print('----------------------Stage 2-------------------------')
        print("cls_f1: {}, exp_f1: {}, exp_p: {}, exp_r: {}\n".format(cls_f1, exp_f1, exp_precision, exp_recall))
        print(".........., att_f1: {}, att_p: {}, att_r: {}\n".format(exp_att_f1, exp_att_p, exp_att_r))

        return cls_f1, exp_f1_p1, exp_att_f1



    def train(self, saved_model_path = None):
        """train model on entire data"""
        exp_labels = self.exp_labels_mapping
        cls_labels = self.cls_labels
        # initialize base model
        self.model = FACBERT(model_config = self.args.model_config, cls_hidden_size = self.args.cls_hidden_size, 
                    exp_hidden_size = self.args.exp_hidden_size, cls_classes = self.cls_classes)
        self.model.to(self.args.device)
        cls_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        exp_criterion = utils.resampling_rebalanced_crossentropy(seq_reduction = 'none')
        
        n_batches = int(np.ceil(len(self.data)/self.args.train_batch_size))
        self.optimizer = AdamW([{'params':[ param for name, param in self.model.named_parameters() if 'cls_layers' not in name]}], lr = self.args.lr, eps = 1e-8)
        total_step = n_batches * self.args.n_epochs
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                num_warmup_steps=0,
                                                                num_training_steps=total_step)

        # phase 1 prediction
        print(">> Phase 1........................................")
        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            train_loss = self.fit(self.input_ids, self.attention_masks, cls_labels, exp_labels, 
                                  cls_criterion, exp_criterion, self.args.train_batch_size, mode = 'exp')
            print("Epoch: %d, train_loss: %.3f, time: %.3f" %(epoch, train_loss, time.time() - begin_time))

        cls_pred_labels, exp_pred_labels_p1, _, exp_pred_train_probs, _, _ = self.predict(self.input_ids, self.attention_masks,
                                                cls_labels, exp_labels, cls_criterion, exp_criterion, self.args.test_batch_size, mode='exp')
        
        exp_pred_train_probs = torch.stack(exp_pred_train_probs, dim = 0).to(self.args.device)
        exp_pred_labels_p1 = utils.max_pooling(exp_pred_labels_p1, self.tokenized_data_slides, self.data)
        exp_true_labels = utils.max_pooling(exp_labels, self.tokenized_data_slides, self.data)
        cls_f1 = f1_score(cls_labels, cls_pred_labels, average = 'macro')
        exp_f1_p1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true_labels, exp_pred_labels_p1)])
        print("++ Training CLS F1: {}, EXP F1: {}".format(cls_f1, exp_f1_p1))

        #phase 2 prediction 
        print(">> Phase 2........................................")
        self.optimizer = AdamW([{'params':[ param for name, param in self.model.named_parameters() if 'exp_layers' not in name]}], lr = self.args.lr, eps = 1e-8)
        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            train_loss = self.fit(self.input_ids, self.attention_masks, self.cls_labels, exp_pred_train_probs, 
                                  cls_criterion, exp_criterion, self.args.train_batch_size, mode = 'cls')
            print("Epoch: %d, train_loss: %.3f, time: %.3f" %(epoch, train_loss, time.time() - begin_time))

        cls_pred_labels, exp_pred_labels_p2, _, _, _, attentions = self.predict(self.input_ids, self.attention_masks,
                                                cls_labels, exp_labels, cls_criterion, exp_criterion, self.args.test_batch_size, mode='exp')
        
        # get top rationale token based on attention weights
        topk_attentions = []
        for i in range(len(exp_pred_labels_p1)):
            k = int(sum(exp_pred_labels_p1[i]))
            if k != 0:
                mink = min(attentions[i].topk(k).values)
                topk_attentions.append(list(np.array(attentions[i].numpy()>=mink.numpy())*1))
            else:
                topk_attentions.append([0 for i in range(len(attentions[i]))])

        exp_attention_pred = utils.max_pooling(topk_attentions, self.tokenized_data_slides, self.data)
        exp_pred_labels_p2 = utils.max_pooling(exp_pred_labels_p2, self.tokenized_data_slides, self.data)
        exp_true_labels = utils.max_pooling(exp_labels, self.tokenized_data_slides, self.data)
        exp_f1_p2 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true_labels, exp_pred_labels_p2)])

        exp_att_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true_labels, exp_attention_pred) if sum(y_true)!=0])
        cls_f1 = f1_score(self.cls_labels, cls_pred_labels, average = 'macro')
        print("++ Training CLS F1: {}, EXP F1: {}, ATT F1: {}".format(cls_f1, exp_f1_p2, exp_att_f1))
        
        self.model.cpu()
        # saved model
        if saved_model_path is not None:
            print("Save model to path: {}".format(saved_model_path))
            torch.save(self.model.state_dict(), saved_model_path)

    # load model
    def load(self, cls_classes = 6, saved_model_path = None):
        if saved_model_path == None:
            print("Please enter the model path...")
            sys.exit(-1)
        try:
            self.model = FACBERT(model_config = self.args.model_config, cls_hidden_size = self.args.cls_hidden_size, 
                    exp_hidden_size = self.args.exp_hidden_size, cls_classes = cls_classes)
            self.model.load_state_dict(torch.load(saved_model_path))
        
        except Exception as e:
            print("Exception")
            print(e)
        
    # classify new data
    def classify(self, new_data):
        """ classify new data """
       
        self.model.to(self.args.device)
        
        data = np.array([self.tokenizer.cls_token+" "+x+ " "+ self.tokenizer.sep_token for x in new_data])
        tokenized_data, input_ids, attention_masks, tokenized_data_slides = \
                        utils.tokenize_text(self.tokenizer, data, self.tokenizer.pad_token)
        
        tokenized_data, input_ids = np.array(tokenized_data, dtype=object), torch.tensor(input_ids, dtype = torch.long)
        attention_masks, tokenized_data_slides= torch.tensor(attention_masks, dtype = torch.long), np.array(tokenized_data_slides, dtype=object)
        
        cls_preds, exp_preds, cls_probs, exp_probs, _, attentions = self.predict(input_ids, attention_masks,
                                                 batch_size = self.args.test_batch_size, mode='exp')
        
 
        exp_preds = utils.max_pooling(exp_preds, tokenized_data_slides, data)
        exp_probs = utils.max_pooling(exp_probs, tokenized_data_slides, data, prob = True)
        attentions = utils.max_pooling(attentions, tokenized_data_slides, data, prob=True)
        
        exp_texts = []
        exp_text_masked = []
        exp_text_probs = []
        att_scores = []
        for prepro_txt, exp_label, exp_prob, att_score in zip(data, exp_preds, exp_probs, attentions):
            try:
                text = prepro_txt.split(" ")
                exp_text_i = ' '.join(text[i] for i in range(1, len(text) - 1) if (exp_label[i]==1))
                exp_text_masked_i = ' '.join(text[i] if (exp_label[i]==1) else "*" for i in range(1, len(text) - 1))
                pred_text_prob_i = ' '.join(text[i]+"§§§"+str(exp_prob[i]) for i in range(1, len(text)-1))
                att_i = ' '.join(text[i]+"§§§"+str(att_score[i]) for i in range(1, len(text)-1))
                exp_texts.append(exp_text_i)
                exp_text_masked.append(exp_text_masked_i)
                exp_text_probs.append(pred_text_prob_i)
                att_scores.append(att_i)
            except Exception as e:
                print("Exception: ...")
                print(e)
        
        return cls_preds, cls_probs, exp_texts, exp_text_masked, att_scores
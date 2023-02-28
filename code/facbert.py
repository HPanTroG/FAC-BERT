import torch.nn as nn
from transformers import AutoModel
import torch


class FACBERT(nn.Module):
    def __init__(self, model_config = 'vinai/bertweet-base', exp_hidden_size = 64, cls_hidden_size = 64, cls_classes = 6):
        super(FACBERT, self).__init__()
        self.model_config = model_config 
        self.baseFAC = AutoModel.from_pretrained(model_config)
        self.exp_hidden_size = exp_hidden_size
        self.cls_hidden_size = cls_hidden_size
        self.cls_classes = cls_classes

        class ExpLayers(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(ExpLayers, self).__init__()
                self.exp_gru = nn.GRU(input_size, hidden_size)
                self.exp_linear = nn.Linear(hidden_size, 1, bias = True)
                self.exp_out = nn.Sigmoid()
            
            def forward(self, inputs):
                return self.exp_out(self.exp_linear(self.exp_gru(inputs)[0]))
        
        class ClsLayers(nn.Module):
            def __init__(self, hidden_size, output_size):
                super(ClsLayers, self).__init__()
                self.v = nn.Parameter(torch.randn((hidden_size, 1)), requires_grad = True)
                self.w = nn. Linear(hidden_size, hidden_size, bias = False)
                self.dropout = nn.Dropout(p = 0.1)
                self.linear = nn.Linear(hidden_size, output_size, bias = True)

            def forward(self, inputs, attention_masks):
                raw_score = self.w(inputs)
                score = torch.tanh(raw_score) @ self.v 
                mask = (attention_masks == 0).unsqueeze(2)
                score[mask] = float('-inf')
                weights = torch.softmax(score, dim = 1)
                context = weights * inputs 
                context = context.sum(dim = 1)
                context = self.dropout(context)
                outputs = self.linear(context)
                return outputs, weights.squeeze()
            
        self.exp_layers = ExpLayers(self.baseFAC.config.hidden_size, self.exp_hidden_size)
        self.cls_layers = ClsLayers(self.baseFAC.config.hidden_size, self.cls_classes)
    
    def forward(self, input_ids, attention_mask):
        '''
            mode: exp: train exp layer only
                  cls: train cls layer only
        '''
        outputs = self.baseFAC(input_ids, attention_mask)
       
        exp_outputs = self.exp_layers(outputs[0]).squeeze() * attention_mask
        cls_outputs, attention_weights = self.cls_layers(outputs[0], attention_mask)
        return cls_outputs, attention_weights, exp_outputs
      
    

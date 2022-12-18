# %% [code]
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn
from torch.utils.data import Dataset

#================
# Dataset
#================
class FeedBackDataset(Dataset):
    def __init__(self, df : pd.DataFrame, tokenizer : AutoTokenizer, max_length : int, target_label : list = None, train_mode : bool = True):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.texts = df['full_text'].values
        self.train_mode = train_mode
        if train_mode:
            self.targets = df[target_label].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.texts[index]
        inputs = self.tokenizer.encode_plus(
                text,
                truncation=True,
                add_special_tokens=True,
                max_length= self.max_len
                )
        
        if self.train_mode :
            return {'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'target': self.targets[index]
                    }
        else:
            return {'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']}

#================
# Loss Funtions
#================
class RMSELoss(nn.Module):
    """
    Code taken from Y Nakama's notebook (https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
    """
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, predictions, targets):
        loss = torch.sqrt(self.mse(predictions, targets) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

def loss_fn(outputs, labels, loss_type='mse'):
    if loss_type == 'mse':
        loss_func = nn.MSELoss()
    elif loss_type == 'rmse':
        loss_func = RMSELoss()
    elif loss_type == 'smooth_l1':
        loss_func = nn.SmoothL1Loss()
    return loss_func(outputs.float(), labels.float())

#================
# Optimizer
#================
def optimizer_setup(model, config : dict, train_dataset_size : int, layerwise : bool = False ):
    
    if layerwise:
        optimizer_parameters = get_optimizer_grouped_parameters(model, config["learning_rate"], config["decoder_lr"], config['weight_decay'])
    else:
        # Define Optimizer and Scheduler
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
                        {
                            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                            "weight_decay": config['weight_decay'],
                        },
                        {
                            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                            "weight_decay": 0.0,
                        },
        ]
    
    optimizer = AdamW(optimizer_parameters, 
                      lr=config['learning_rate'],
                      correct_bias = True
                     )
    num_training_steps = (train_dataset_size * config['epochs']) // (config['train_batch_size'] * config['n_accumulate'])
    
    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0.1*num_training_steps,
                num_training_steps=num_training_steps
            )
    return optimizer, scheduler

#Layerwise Learning Rate Decay
def get_optimizer_grouped_parameters(model, 
                                     layerwise_lr : float,
                                     decoder_lr : float,
                                     layerwise_weight_decay : float = 0.0,
                                     layerwise_lr_decay : float =0.9 ):

    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if "model" not in n],
                                        "weight_decay": 0.0,
                                        "lr": decoder_lr,
                                    },]
    
    # initialize lr for extra params in encoder
    extra_params = [(n,p) for n, p in model.named_parameters() if "model" in n and "model.embeddings" not in n and "model.encoder.layer" not in n]
    optimizer_grouped_parameters += [{"params": [p for n, p in extra_params if not any(nd in n for nd in no_decay)],
                                        "weight_decay": layerwise_weight_decay,
                                        "lr": layerwise_lr,
                                    },]
    optimizer_grouped_parameters += [{"params": [p for n, p in extra_params if any(nd in n for nd in no_decay)],
                                        "weight_decay": 0.0,
                                        "lr": layerwise_lr,
                                    },]
    
    # initialize lrs for every layer
    layers = [model.model.embeddings] + list(model.model.encoder.layer)
    layers.reverse()
    lr = layerwise_lr
    for layer in layers:
        optimizer_grouped_parameters += [{"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                                            "weight_decay": layerwise_weight_decay,
                                            "lr": lr,
                                            },
                                            {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                                            "weight_decay": 0.0,
                                            "lr": lr,
                                            },]
        lr *= layerwise_lr_decay
    return optimizer_grouped_parameters

#================
# Evaluation
#================
def compute_metrics(p):
    predictions = p.predictions
    labels = p.label_ids
    colwise_rmse = np.sqrt(np.mean((labels - predictions) ** 2, axis=0))
    mean_rmse = np.mean(colwise_rmse)
    return {"mcrmse": mean_rmse,
            'cohesion_rmse': colwise_rmse[0],
            'syntax_rmse': colwise_rmse[1],
            'vocabulary_rmse': colwise_rmse[2],
            'phraseology_rmse': colwise_rmse[3],
            'grammar_rmse': colwise_rmse[4],
            'conventions_rmse': colwise_rmse[5]}

# -*- coding: utf-8 -*-
"""transformers_summarization_works!.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10dRhpha5SLvREjgQWdmoGMzStwV1fb2F
"""

!pip install -q transformers
!pip install -q sentencepiece

"""*Checking out the GPU we have access to*"""

!nvidia-smi

"""*Imports*"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration

"""*Mount drive*"""

from google.colab import drive
drive.mount('/content/drive')

"""*Setting up the device for GPU usage*"""

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

df = pd.read_csv("/content/drive/MyDrive/Dissertation/data/pickled_for_colab.csv",encoding='latin-1')

"""*Setting constants*"""

MAX_LEN = 512
EPOCHS = 2
SUMMARY_LEN = 144
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2

"""*DataHandler: reads in the dataset and formats it*"""

def remove_by_indices(descr, indxs):
  return [e for i, e in enumerate(descr) if i not in indxs]

class DataHandler(Dataset):

    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.episode_description = self.data.episode_description
        self.transcript = self.data.transcript

    def __len__(self):
        return len(self.episode_description)

    def __getitem__(self, index):

        _transcript = self.episode_description.tolist()
        _description = self.transcript.tolist()

        indxs = []

        for i, des in enumerate(_transcript):
          if not isinstance(des, str):
            indxs.append(i)

        for i, des in enumerate(_description):
          if not isinstance(des, str):
            indxs.append(i)

        episode_description = remove_by_indices(_transcript, indxs)
        transcript = remove_by_indices(_description, indxs)

        transcript = str(self.transcript[index])
        transcript = ' '.join(transcript.split())

        episode_description = str(self.text[index])
        episode_description = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([transcript], max_length= MAX_LEN, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([episode_description], max_length= SUMMARY_LEN, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

"""*Training model*"""

def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=labels)
        loss = outputs[0]
        
        if i%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

"""*Validating model*"""

def validate(epoch):
    model.eval()
  
    predictions = []
    actuals = []

    with torch.no_grad():
        for i, data in enumerate(testing_loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            gens = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in gens]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            
            predictions.extend(preds)
            actuals.extend(target)

            if i%100==0:
                print(f'Completed {_}')
    return predictions, actuals

"""*Initialisation of the tokeniser, model and datasets*"""

t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state=42).reset_index(drop=True)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)

training_set = DataHandler(train_dataset, bart_tokenizer)
testing_set = DataHandler(test_dataset, bart_tokenizer)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
model = bart_model.to(device)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=1e-4)

"""*Generating the summaries and writing to csv*"""

for epoch in range(2):
    train(epoch)

for epoch in range(1):
    predictions, actuals = validate(epoch)
    output_df = {'Summary': actuals,
        'Predicted Summary': predictions       
        }
    output_df.to_csv('/content/drive/MyDrive/Dissertation/data/predictions_bart_textrank.csv')
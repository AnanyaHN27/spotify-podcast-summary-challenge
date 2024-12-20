# -*- coding: utf-8 -*-
"""bert_k_means

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HI17UI2jyGTAWGB_11ZKNBVR4wUK6_H1
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers

!pip install -q sentencepiece

# Checking out the GPU we have access to
!nvidia-smi

# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertModel

# # Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

df = pd.read_csv("/content/drive/MyDrive/Dissertation/data/pickled_for_colab.csv",encoding='latin-1')

df.head()

# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
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

        source = self.tokenizer.batch_encode_plus([transcript], max_length= 512, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([episode_description], max_length= 144, pad_to_max_length=True,return_tensors='pt')

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

def train(epoch):
    model.train()
    for i,data in enumerate(training_loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask)
        loss = outputs[0]
        
        if i%500==0:
            print(f'Epoch: {epoch}, Loss:  {(loss)}')
        
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

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

            
    return predictions, actuals

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
EPOCHS = 2

train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state=42).reset_index(drop=True)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)

training_set = DataHandler(train_dataset, tokenizer)
testing_set = DataHandler(test_dataset, tokenizer)

training_loader = DataLoader(training_set, batch_size= 2, shuffle= True)
testing_loader = DataLoader(testing_set, batch_size= 2, shuffle= False)

bert_model = BertModel.from_pretrained('bert-base-cased')
model = bert_model.to(device)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=1e-4)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def k_means(sentence_embedding_list):
  clusters = int(np.ceil(len(sentence_embedding_list)**0.5))
  kmeans = KMeans(n_clusters=clusters).fit(sentence_embedding_list)
  #minimum distances from each sentence cluster to each sentence in the embedding list
  sum_index, _ = sorted(pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_embedding_list,metric='euclidean'))
  return sum_index

def bert_summarise(sum_index, text):
  sentences = sent_tokenize(text)
  sentence_embedding_list = bert_sent_embedding(sentences)
  sum_index = k_means(sentence_embedding_list)
  summary = ' '.join([sentences[ind] for ind in sum_index])
  return summary

for epoch in range(2):
    train(epoch)

sentence_embedding_list = mean_pooling(model(**tokenizer(testing_set)))

sum_of_squared_distances = []
for k in len(sentence_embedding_list):
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    sum_of_squared_distances.append(km.inertia_)
plt.plot(sentence_embedding_list, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

sum_index = [k_means(embedding) for embedding in sentence_embedding_list]
summ_list = [bert_summarise(sum_index[i], t) for enumerate(i, t) in sentence_embedding_list]

(pd.DataFrame(summ_list)).to_csv("/content/drive/MyDrive/Dissertation/data/bert_k_means.csv", index=False)

(pd.DataFrame(summ_list)).to_csv("/content/drive/MyDrive/Dissertation/data/bert_k_means.csv", index=False)

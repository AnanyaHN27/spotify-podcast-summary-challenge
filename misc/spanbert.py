# -*- coding: utf-8 -*-
"""spanbert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GVq8qMTCazUIR12MNtf_rjeY7mtfea13
"""

!pip install transformers
!pip install rouge
!pip install neuralcoref==4.0
!pip install spacy==2.1.0 #notebook crashes on other versions
!python -m spacy download en
!pip install bert-extractive-summarizer

!pip install spacy
!pip install neuralcoref
!pip install --upgrade bert-extractive-summarizer

import pandas as pd
import csv
reference_results = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/pickled_for_colab.csv')
reference_results.head(1)

import torch
from summarizer import Summarizer
model = Summarizer()
from summarizer import Summarizer
#from summarizer.coreference_handler import CoreferenceHandler
from transformers import AutoConfig, AutoTokenizer, AutoModel

custom_config = AutoConfig.from_pretrained("SpanBERT/spanbert-base-cased")
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
custom_model = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased", config=custom_config)

spanbert_model = Summarizer(custom_model = custom_model, custom_tokenizer=custom_tokenizer)

def spanBert(transcript, num_sentences):
  result = spanbert_model(transcript, min_length=60, num_sentences=num_sentences)
  return ''.join(result)

end = 0
new_df = []
for i in range(transcripts.shape[0]):
  print("step: ", i)
  summ = spanBert(transcripts['transcript'][i], 4)
  new_df.append([summ])
  if (end%50 == 0):
    df = pd.DataFrame(new_df,columns=['trans'])

    name = "span_bert" + "_" + str(i) + ".csv"
    
    df.to_csv('/content/drive/MyDrive/Colab Notebooks/' + name)
  end += 1

#transcripts['spanBert_t5'] = transcripts.apply(lambda row: spanBert(row['transcript'], 15), axis=1)
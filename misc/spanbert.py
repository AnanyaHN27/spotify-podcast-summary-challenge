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

"""*Imports*"""

import pandas as pd
import csv
import torch
from summarizer import Summarizer
from summarizer.coreference_handler import CoreferenceHandler
from transformers import AutoConfig, AutoTokenizer, AutoModel

"""*Define the configurations, model and tokenizer*"""

config = AutoConfig.from_pretrained("SpanBERT/spanbert-base-cased")
config.output_hidden_states=True
tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
model = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased", config=custom_config)
handler = CoreferenceHandler(greedyness=.5)
spanbert_model = Summarizer(custom_model = model, custom_tokenizertokenizer, sentence_handler=handler)

"""*Train SpanBERT*"""

def spanBert(transcript):
  #inputs = tokenizer(transcript, max_length=512, pad_to_max_length=True, return_tensors="pt")
  outputs = spanbert_model(transcripts, ratio=0.3)
  return ''.join(outputs)

"""*Generate and write summaries to csv to be used as input for others*"""

reference_results = pd.read_csv('/content/drive/MyDrive/Dissertation/data/pickled_for_colab.csv')

summaries_df = []
for i in range(transcripts.shape[0]):
  print("step: ", i)
  summ = spanBert(transcripts['transcript'][i])
  summaries_df.append([summ])

df.to_csv('/content/drive/MyDrive/Dissertation/data/' + 'span_bert.csv')

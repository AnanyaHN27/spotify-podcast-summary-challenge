# -*- coding: utf-8 -*-
"""perplexity.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IMnvqtjkDOvE_tt4dgPW2URLZuu9DImf
"""

!pip install -U nltk

import nltk
nltk.download('punkt')
import pandas as pd
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary
import os

#train_sentences = pd.read_csv("/content/drive/MyDrive/Dissertation/data/pickled_for_colab.csv").episode_description[:500].tolist()
train_sentences = []

for filename in os.listdir(os.path.expanduser('~/coca-samples-text')):
    if filename.endswith("txt"): 
        with open(filename) as f:
          train_sentences.extend(f.readlines())

tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]

n = 2
train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
words = [word for sent in tokenized_text for word in sent]
words.extend(["<s>", "</s>"])
padded_vocab = Vocabulary(words)
model = Laplace(n)
model.fit(train_data, padded_vocab)

test_sentences = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/longformer_results (1).csv").summaries.tolist()

#test_sentences = ['an apple', 'an ant', 'Here is a second article that Ive written. While the content is the same, the stylistic choices that I am making are quite different from the ones I made upon crafting the first article used in training.']
tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]

test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]

tot = 0
for i, test in enumerate(test_data):
  tot += model.perplexity(test)
print(tot/len(test_data))
"""*Imports*"""
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import numpy as np

"""*Get tokeniser and the data*"""
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
df = pd.read_csv("/data/pickled_for_colab.csv")

trans_lst = df.transcript.tolist()

encoded_final_lst = []

"""*Handle the truncation*"""
for trans in trans_lst:
  enc_lst = tokenizer.encode(trans)
  if len(enc_lst) > 512:

    ##truncating after the last full stop in the first 256 characters
    first_half = enc_lst[:256].split(".")
    first_half = (".".join(first_half) + ".") if first_half[-1][-1] in [".", "!"] else (".".join(first_half[:-1]) + ".")

    ##truncating after the first full stop in the first 256 characters
    second_half = enc_lst[len(samp)-256:].split(".")
    second_half = (".".join(second_half)) if second_half[0][0].isupper() else ".".join(second_half[1:])
    
    truncated = first_half + second_half
    encoded_final_lst.append([truncated])

"""*Write to the file*"""
encoded_final_lst.to_csv("/data/custom_truncation.csv")

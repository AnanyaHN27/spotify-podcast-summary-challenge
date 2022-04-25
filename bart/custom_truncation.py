from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import numpy as np

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
df = pd.read_csv("/data/pickled_for_colab.csv")

trans_lst = df.transcript.tolist()

encoded_final_lst = []

for trans in trans_lst:
  enc_lst = tokenizer.encode(trans)
  if len(enc_lst) > 512:

    first_half = enc_lst[:256].split(".")
    first_half = (".".join(first_half) + ".") if first_half[-1][-1] in [".", "!"] else (".".join(first_half[:-1]) + ".")

    second_half = enc_lst[len(samp)-256:].split(".")
    second_half = (".".join(second_half)) if second_half[0][0].isupper() else ".".join(second_half[1:])
    
    truncated = first_half + second_half
    encoded_final_lst.append([truncated])

encoded_final_lst.to_csv("/data/custom_truncation.csv")

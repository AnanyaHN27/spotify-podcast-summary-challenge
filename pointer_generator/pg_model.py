# -*- coding: utf-8 -*-
"""pg_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FCLJHdMNZ8Vf9O_YzFS1bJ3iQwNfrWSj

*Imports*
"""

import os
import re
from tempfile import TemporaryDirectory
import subprocess
from multiprocessing.dummy import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
import random
from random import shuffle
from functools import lru_cache
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rouge import Rouge
import math
from tqdm import tqdm
import pandas as pd
import tarfile
from io import BytesIO

"""*Setting the device*"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-31

"""*Vocab and OOV order*"""

class Vocab(object):

  PAD = 0
  SOS = 1
  EOS = 2
  UNK = 3


  def __init__(self):
    self.word2index = {}
    self.word2count = Counter()
    self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    self.index2word = self.reserved[:]
    self.embeddings = []

  def add_words(self, words):
    for word in words.split(" "):
      if word not in self.word2index:
        self.word2index[word] = len(self.index2word)
        self.index2word.append(word)
    self.word2count.update(words)

  def load_embeddings(self, file_path="/content/drive/MyDrive/Dissertation/data/textrank.csv"):
    num_embeddings = 0
    vocab_size = len(self)
    with open(file_path, encoding='utf-8') as f:
      for line in f:
        line = line.split()
        idx = self.word2index.get(line[0])
        if idx is not None: 
          coefs = np.array(line[1:], dtype='float32')
          self.embeddings[idx] = coefs
          num_embeddings += 1
    return num_embeddings

  def __getitem__(self, item):
    return self.word2index.get(item, self.UNK)

  def __len__(self):
    return len(self.index2word)

class OOVDict(object):

  def __init__(self, base_oov_idx):
    self.word2index = {} 
    self.index2word = {}  
    self.next_index = {}  
    self.base_oov_idx = base_oov_idx
    self.ext_vocab_size = base_oov_idx

  def add_word(self, idx_in_batch, word):
    key = (idx_in_batch, word)
    index = self.word2index.get(key)
    if index is None: 
      index = self.next_index.get(idx_in_batch, self.base_oov_idx)
      self.next_index[idx_in_batch] = index + 1
      self.word2index[key] = index
      self.index2word[(idx_in_batch, index)] = word
      self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
    return index

"""*Handling the dataset*"""

class Dataset(object):
  
  def __init__(self, filename):
    self.filename = filename
    self.pairs = []
    self.src_len = 0
    self.tgt_len = 0
    self.word_to_index = {}

    df = pd.read_csv(self.filename) 

    self.transcript = df.transcript
    self.episode_description = df.episode_description
    
    _transcript = self.episode_description.tolist()
    _description = self.transcript.tolist()

    indxs = []

    for i, des in enumerate(_transcript):
      if not isinstance(des, str):
        indxs.append(i)

    for i, des in enumerate(_description):
      if not isinstance(des, str):
        indxs.append(i)

    self.episode_description = self.remove_by_indices(_transcript, indxs)
    self.transcript = self.remove_by_indices(_description, indxs)
    self.pairs = list(zip(self.transcript, self.episode_description))

    self.src_len = len(self.transcript) + 1  # EOS
    self.tgt_len = len(self.episode_description) + 1  # EOS
  
  def remove_by_indices(self, descr, indxs):
    return [e for i, e in enumerate(descr) if i not in indxs]

  def build_vocab(self, src=True, tgt=True,
                  embed_file="/content/drive/MyDrive/Dissertation/data/textrank.csv"):
    
    vocab = Vocab()
    for example in self.pairs:
      if src:
        vocab.add_words(example[0]) #takes sentence as input
      if tgt:
        vocab.add_words(example[1]) #takes sentence as input

    updated_ind2word = []

    for i in vocab.index2word:
      if len(i) != 1:
        updated_ind2word.append(i)

    updated_word2ind = {}
    for k,v in vocab.word2index.items():
      if len(k) != 1:
        updated_word2ind[k] = v

    vocab.ind2word = updated_ind2word
    vocab.word2index = updated_word2ind

    print("\n word2index", vocab.word2index)
    print("\n index2word", vocab.index2word)

    return vocab

  def generator(self, batch_size = 1, src_vocab=None, tgt_vocab=None, ext_vocab=True):
    ptr = len(self.pairs)  # make sure to shuffle at first run
    if ext_vocab:
      assert src_vocab is not None
      base_oov_idx = len(src_vocab)
    while True:
      if ptr + batch_size > len(self.pairs):
        random.shuffle(self.pairs)  
        ptr = 0
      examples = self.pairs[ptr:ptr + batch_size]
      ptr += batch_size
      src_tensor, tgt_tensor = None, None
      lengths, oov_dict = None, None
      if src_vocab or tgt_vocab:
        # initialize tensors
        if src_vocab:
          examples.sort(key=lambda x: -len(x[0]))
          lengths = [len(x[0]) for x in examples]
          max_src_len = lengths[0]
          src_tensor = torch.zeros(max_src_len, batch_size, dtype=torch.long)
          if ext_vocab:
            oov_dict = OOVDict(base_oov_idx)
        
        if tgt_vocab:
          max_tgt_len = max(len(x[1]) for x in examples)
          tgt_tensor = torch.zeros(max_tgt_len, batch_size, dtype=torch.long)
        # fill up tensors by word indices
        for i, example in enumerate(examples):
          if src_vocab:
            for j, word in enumerate(example[0]):
              idx = src_vocab[word]
              if ext_vocab and idx == src_vocab.UNK:
                idx = oov_dict.add_word(i, word)
              src_tensor[j, i] = idx
            src_tensor[len(example[0]) - 1, i] = src_vocab.EOS
          if tgt_vocab:
            for j, word in enumerate(example[1]):
              idx = tgt_vocab[word]
              if ext_vocab and idx == src_vocab.UNK:
                idx = oov_dict.word2index.get((i, word), idx)
              tgt_tensor[j, i] = idx
            tgt_tensor[len(example[1]) - 1, i] = tgt_vocab.EOS

      batch = {
          "examples": examples,
          "input": src_tensor,
          "target": tgt_tensor,
          "inp_lengths": lengths,
          "oov_dict": oov_dict
      }
      yield batch

"""*Code for training*"""

def train_batch(batch, model, criterion, optimizer, *,
                forcing_ratio=0.5, partial_forcing=True, sample=False,
                rl_ratio=0.5, vocab=None, show_cover_loss=False):
  input_lengths = batch["inp_lengths"]

  optimizer.zero_grad()
  input_tensor = batch["input"].to(DEVICE)
  target_tensor = batch["target"].to(DEVICE)
  ext_vocab_size = batch["oov_dict"].ext_vocab_size

  out = model(input_tensor, target_tensor, input_lengths, criterion,
              forcing_ratio=forcing_ratio, partial_forcing=partial_forcing, sample=sample,
              ext_vocab_size=ext_vocab_size, include_cover_loss=show_cover_loss)

  if rl_ratio>0:
    samp = model(input_tensor, saved_out=out, criterion=criterion, sample=True,
                       ext_vocab_size=ext_vocab_size)
    #this is a randomly sampled output
    baseline = model(input_tensor, saved_out=out, ext_vocab_size=ext_vocab_size)
    #this is the greedy baseline generated by the model
    scores = eval_batch_output([ex.tgt for ex in batch["examples"]], vocab, batch["oov_dict"],
                               samp.decoded_tokens, baseline.decoded_tokens)
    greedy_rouge = scores[1]['f']
    neg_reward = greedy_rouge - scores[0]['f']
    # if sample > baseline, the reward is positive and rl_loss is negative
    rl_loss = neg_reward * samp.loss
    rl_loss_value = neg_reward * samp.loss_value
    loss = (1 - rl_ratio) * out.loss + rl_ratio * rl_loss
    loss_value = (1 - rl_ratio) * out.loss_value + rl_ratio * rl_loss_value
  else:
    loss = out.loss
    loss_value = out.loss_value

  loss.backward()
  optimizer.step()

  target_length = target_tensor.size(0)
  return loss_value / target_length, None

def train(train_generator, vocab, model, valid_generator=None, rl_ratio=1
      ,saved_state=None):
  
  model.to(DEVICE)
  
  optimizer = optim.Adagrad(model.parameters(), lr=0.001,
                                initial_accumulator_value=0.1)
    
  past_epochs = 10
  rl_start_epoch = 2
  total_batch_count = 0
  
  criterion = nn.NLLLoss(ignore_index=vocab.PAD)
  best_avg_loss = float("inf")
  
  best_epoch_id = None

  for epoch_count in range(1 + past_epochs,  100):
    if epoch_count >= rl_start_epoch:
      rl_ratio = 1
    else:
      rl_ratio = 0
    epoch_loss, epoch_metric = 0, 0
    epoch_avg_loss, valid_avg_loss, valid_avg_metric = None, None, None
    prog_bar = tqdm(range(1, 1000 + 1), desc='Epoch %d' % epoch_count)
    model.train()

    for batch_count in prog_bar:  # training batches
     
      forcing_ratio = 0.75  * 0.9999 / (
                  0.9999 + np.exp(total_batch_count / 0.9999))

      batch = next(train_generator)
      loss, metric = train_batch(batch, model, criterion, optimizer,
                                 forcing_ratio=forcing_ratio,
                                 partial_forcing=True, sample=True,
                                 vocab=vocab,
                                 show_cover_loss=False)

      epoch_loss += float(loss)
      epoch_avg_loss = epoch_loss / batch_count
      if metric is not None: 
        epoch_metric += metric
        epoch_avg_metric = epoch_metric / batch_count
        prog_bar.set_postfix(loss='%g' % epoch_avg_loss, rouge='%.4g' % (epoch_avg_metric * 100))
      else:
        prog_bar.set_postfix(loss='%g' % epoch_avg_loss)

    filename = '%s.%02d.pt' % ("/content/drive/MyDrive/Dissertation/data/afresh", epoch_count)
    torch.save(model, filename)
  
    if epoch_avg_loss < best_avg_loss:
      best_epoch_id = epoch_count
      best_avg_loss = epoch_avg_loss

    for epoch_id in range(1 + past_epochs, epoch_count):
      if epoch_id != best_epoch_id:
        try:
          prev_filename = '%s.%02d.pt' % ("model_path_prefix", epoch_id)
          os.remove(prev_filename)
        except FileNotFoundError:
          pass
          
      # save training status
      torch.save({
        'epoch': epoch_count,
        'total_batch_count': total_batch_count,
        'train_avg_loss': epoch_avg_loss,
        'valid_avg_loss': valid_avg_loss,
        'valid_avg_metric': valid_avg_metric,
        'best_epoch_so_far': best_epoch_id,
        'optimizer': optimizer
      }, '%s.train.pt' % "model_path_prefix")

    if rl_ratio > 0:
      rl_ratio **= rl_ratio_power

"""*Creating the hypothesis for potential summaries*"""

class Hypothesis(object):

  def __init__(self, tokens, log_probs, dec_hidden, dec_states, enc_attn_weights, num_non_words):
    self.tokens = tokens  
    self.log_probs = log_probs  
    self.dec_hidden = dec_hidden  
    self.dec_states = dec_states  
    self.enc_attn_weights = enc_attn_weights  
    self.num_non_words = num_non_words 

  def __repr__(self):
    return repr(self.tokens)

  def __len__(self):
    return len(self.tokens) - self.num_non_words

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.log_probs)

  def create_next(self, token, log_prob, dec_hidden, add_dec_states, enc_attn, non_word):
    return Hypothesis(tokens=self.tokens + [token], log_probs=self.log_probs + [log_prob],
                      dec_hidden=dec_hidden, dec_states=
                      self.dec_states + [dec_hidden] if add_dec_states else self.dec_states,
                      enc_attn_weights=self.enc_attn_weights + [enc_attn],
                      num_non_words=self.num_non_words + 1 if non_word else self.num_non_words)

"""*Creating the Encoder, Decoder, Seq2Seq models*"""

class EncoderRNN(nn.Module):

  def __init__(self, embed_size, hidden_size, bidi=True, *, rnn_drop=0):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_directions = 2 if bidi else 1
    self.gru = nn.GRU(embed_size, hidden_size, bidirectional=bidi, dropout=rnn_drop)

  def forward(self, embedded, hidden, input_lengths=None):
    
    if input_lengths is not None:
      embedded = pack_padded_sequence(embedded, input_lengths)

    output, hidden = self.gru(embedded, hidden)

    if input_lengths is not None:
      output, _ = pad_packed_sequence(output)

    if self.num_directions > 1:
      batch_size = hidden.size(1)
      hidden = hidden.transpose(0, 1).contiguous().view(1, batch_size,
                                                        self.hidden_size * self.num_directions)
    return output, hidden

  def init_hidden(self, batch_size):
    return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=DEVICE)


class DecoderRNN(nn.Module):

  def __init__(self, vocab_size, embed_size, hidden_size, *, enc_attn=True, dec_attn=True,
               enc_attn_cover=True, pointer=True, tied_embedding=None, out_embed_size=None,
               in_drop=0, rnn_drop=0, out_drop=0, enc_hidden_size=None):
    super(DecoderRNN, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.combined_size = self.hidden_size
    self.enc_attn = enc_attn
    self.dec_attn = dec_attn
    self.enc_attn_cover = enc_attn_cover
    self.pointer = pointer
    self.out_embed_size = out_embed_size

    self.in_drop = nn.Dropout(in_drop) if in_drop > 0 else None
    self.gru = nn.GRU(embed_size, self.hidden_size, dropout=rnn_drop)

    if enc_attn:
      if not enc_hidden_size: enc_hidden_size = self.hidden_size
      self.enc_bilinear = nn.Bilinear(self.hidden_size, enc_hidden_size, 1)
      self.combined_size += enc_hidden_size
      if enc_attn_cover:
        self.cover_weight = nn.Parameter(torch.rand(1))

    if dec_attn:
      self.dec_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
      self.combined_size += self.hidden_size

    self.out_drop = nn.Dropout(out_drop) if out_drop > 0 else None
    if pointer:
      self.ptr = nn.Linear(self.combined_size, 1)

    if tied_embedding is not None and embed_size != self.combined_size:
      # use pre_out layer if combined size is different from embedding size
      self.out_embed_size = embed_size

    if self.out_embed_size:  # use pre_out layer
      self.pre_out = nn.Linear(self.combined_size, self.out_embed_size)
      size_before_output = self.out_embed_size
    else:  # don't use pre_out layer
      size_before_output = self.combined_size

    self.out = nn.Linear(size_before_output, vocab_size)
    if tied_embedding is not None:
      self.out.weight = tied_embedding.weight

  def forward(self, embedded, hidden, encoder_states=None, decoder_states=None, coverage_vector=None, *,
              encoder_word_idx=None, ext_vocab_size=None, log_prob=True):
    batch_size = embedded.size(0)
    combined = torch.zeros(batch_size, self.combined_size, device=DEVICE)

    if self.in_drop: embedded = self.in_drop(embedded)

    output, hidden = self.gru(embedded.unsqueeze(0), hidden) 
    combined[:, :self.hidden_size] = output.squeeze(0)        
    offset = self.hidden_size
    enc_attn, prob_ptr = None, None

    if self.enc_attn or self.pointer:
      num_enc_steps = encoder_states.size(0)
      enc_total_size = encoder_states.size(2)
      enc_energy = self.enc_bilinear(hidden.expand(num_enc_steps, batch_size, -1).contiguous(),
                                     encoder_states)
      enc_energy += self.cover_weight * torch.log(coverage_vector.transpose(0, 1).unsqueeze(2) + eps)
      # transpose => (batch size, num encoder states, 1)
      enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)
      enc_context = torch.bmm(encoder_states.permute(1, 2, 0), enc_attn)
      combined[:, offset:offset+enc_total_size] = enc_context.squeeze(2)
      offset += enc_total_size
      enc_attn = enc_attn.squeeze(2)

    if decoder_states is not None and len(decoder_states) > 0:
      dec_energy = self.dec_bilinear(hidden.expand_as(decoder_states).contiguous(),
                                      decoder_states)
      dec_attn = F.softmax(dec_energy, dim=0).transpose(0, 1)
      dec_context = torch.bmm(decoder_states.permute(1, 2, 0), dec_attn)
      combined[:, offset:offset + self.hidden_size] = dec_context.squeeze(2)
    offset += self.hidden_size

    if self.out_drop: 
      combined = self.out_drop(combined)

    # generator
    if self.out_embed_size:
      out_embed = self.pre_out(combined)
    else:
      out_embed = combined
    logits = self.out(out_embed)  # (batch size, vocab size)

    # pointer
    if self.pointer:
      output = torch.zeros(batch_size, ext_vocab_size, device=DEVICE)
      # distribute probabilities between generator and pointer
      prob_ptr = F.sigmoid(self.ptr(combined))  
      prob_gen = 1 - prob_ptr
      # add generator probabilities to output
      gen_output = F.softmax(logits, dim=1)  
      output[:, :self.vocab_size] = (prob_gen * gen_output) + ((1-prob_gen)*enc_attn)
      # add pointer probabilities to output
      if log_prob: output = torch.log(output + eps)
    else:
      if log_prob: output = F.log_softmax(logits, dim=1)
      else: output = F.softmax(logits, dim=1)

    return output, hidden, enc_attn, prob_ptr


class Seq2SeqOutput(object):

  def __init__(self, encoder_outputs, encoder_hidden, decoded_tokens, loss=0,
              loss_value=0, enc_attn_weights=None, ptr_probs=None):
    self.encoder_outputs = encoder_outputs
    self.encoder_hidden = encoder_hidden
    self.decoded_tokens = decoded_tokens  
    self.loss = loss  # scalar
    self.loss_value = loss_value  
    self.enc_attn_weights = enc_attn_weights
    self.ptr_probs = ptr_probs 

class Seq2Seq(nn.Module):

  def __init__(self, vocab, max_dec_steps=None):
    super(Seq2Seq, self).__init__()
    self.vocab = vocab
    self.vocab_size = len(vocab)
    if vocab.embeddings is not None:
      self.embed_size = vocab.embeddings.shape[1]
    self.embed_size = 100
    embedding_weights = None
    self.max_dec_steps = 144 + 1 if max_dec_steps is None else max_dec_steps
    self.enc_attn = True
    self.enc_attn_cover = True
    self.dec_attn = False
    self.pointer = True
    self.cover_loss = 1
    enc_total_size = 150 * 2

    dec_hidden_size = 200  
    self.enc_dec_adapter = nn.Linear(enc_total_size, dec_hidden_size)

    self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=vocab.PAD,
                                  _weight=embedding_weights)
    self.encoder = EncoderRNN(self.embed_size, 150, True,
                              rnn_drop=0)
    self.decoder = DecoderRNN(self.vocab_size, self.embed_size, dec_hidden_size,
                              enc_attn=True, dec_attn=False,
                              pointer=True, out_embed_size=None,
                              tied_embedding=self.embedding,
                              in_drop=0, rnn_drop=0,
                              out_drop=0, enc_hidden_size=enc_total_size)

  def filter_oov(self, tensor, ext_vocab_size):
    #Replace OOV with UNK
    if ext_vocab_size and ext_vocab_size > self.vocab_size:
      result = tensor.clone()
      result[tensor >= self.vocab_size] = self.vocab.UNK
      return result
    return tensor

  def forward(self, input_tensor, target_tensor=None, input_lengths=None, criterion=None, *,
              forcing_ratio=0, partial_forcing=True, ext_vocab_size=None, sample=False,
              saved_out=None, include_cover_loss=False):
   
    input_length = input_tensor.size(0)
    batch_size = input_tensor.size(1)
    log_prob = not (sample or self.decoder.pointer)     
    target_length = self.max_dec_steps
   
    if forcing_ratio == 1:
      use_teacher_forcing = True
    elif forcing_ratio > 0:
      if partial_forcing:
        use_teacher_forcing = None 
      else:
        use_teacher_forcing = random.random() < forcing_ratio
    else:
      use_teacher_forcing = False

    if saved_out:  # reuse encoder states of a previous run
      encoder_outputs = saved_out.encoder_outputs
      encoder_hidden = saved_out.encoder_hidden
      assert input_length == encoder_outputs.size(0)
      assert batch_size == encoder_outputs.size(1)
    else:  # run the encoder
      encoder_hidden = self.encoder.init_hidden(batch_size)
      # encoder_embedded: (input len, batch size, embed size)
      encoder_embedded = self.embedding(self.filter_oov(input_tensor, ext_vocab_size))
      encoder_outputs, encoder_hidden = \
        self.encoder(encoder_embedded, encoder_hidden, input_lengths)

    # initialize return values
    r = Seq2SeqOutput(encoder_outputs, encoder_hidden,
                      torch.zeros(target_length, batch_size, dtype=torch.long))
    
    decoder_input = torch.tensor([self.vocab.SOS] * batch_size, device=DEVICE)
    if self.enc_dec_adapter is None:
      decoder_hidden = encoder_hidden
    else:
      decoder_hidden = self.enc_dec_adapter(encoder_hidden)
    decoder_states = []
    enc_attn_weights = []

    for di in range(target_length):
      decoder_embedded = self.embedding(self.filter_oov(decoder_input, ext_vocab_size))
      if enc_attn_weights:
        coverage_vector, _ = torch.sum(torch.cat(enc_attn_weights), dim=0)
      else:
        coverage_vector = None
      decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = \
        self.decoder(decoder_embedded, decoder_hidden, encoder_outputs,
                     torch.cat(decoder_states) if decoder_states else None, coverage_vector,
                     encoder_word_idx=input_tensor, ext_vocab_size=ext_vocab_size,
                     log_prob=log_prob)
        
      #dec_enc_attn: new attention distribution
      
      if self.dec_attn:
        decoder_states.append(decoder_hidden)
      # save the decoded tokens
      if not sample:
        _, top_idx = decoder_output.data.topk(1)  
      else:
        prob_distribution = torch.exp(decoder_output) if log_prob else decoder_output
        top_idx = torch.multinomial(prob_distribution, 1)
      top_idx = top_idx.squeeze(1).detach()
      r.decoded_tokens[di] = top_idx
      # compute loss
      if criterion:
        if target_tensor is None:
          gold_standard = top_idx 
        else:
          gold_standard = target_tensor[di]
        if not log_prob:
          decoder_output = torch.log(decoder_output + eps)  
        nll_loss = criterion(decoder_output, gold_standard)
        r.loss += nll_loss
        r.loss_value += nll_loss.item()
      # update attention history and compute coverage loss
      if self.enc_attn_cover or (criterion and self.cover_loss > 0):
        if coverage_vector is not None and criterion and self.cover_loss > 0:
          coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn))
          r.loss += coverage_loss
          if include_cover_loss: r.loss_value += coverage_loss.item()
        enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
      # decide the next input
      if use_teacher_forcing or (use_teacher_forcing is None and random.random() < forcing_ratio):
        decoder_input = target_tensor[di]  
        # teacher forcing
      else:
        decoder_input = top_idx
    
    return r

  def beam_search(self, input_tensor, input_lengths=None, ext_vocab_size=None, beam_size=10, *,
                  min_out_len=1, max_out_len=None, len_in_words=True):
    batch_size = input_tensor.size(1)
    assert batch_size == 1
    if max_out_len is None:
      max_out_len = self.max_dec_steps - 1  

    # encode
    encoder_hidden = self.encoder.init_hidden(batch_size)
    encoder_embedded = self.embedding(self.filter_oov(input_tensor, ext_vocab_size))
    encoder_outputs, encoder_hidden = self.encoder(encoder_embedded, encoder_hidden, input_lengths)
    decoder_hidden = self.enc_dec_adapter(encoder_hidden)
    encoder_outputs = encoder_outputs.expand(-1, beam_size, -1).contiguous()
    input_tensor = input_tensor.expand(-1, beam_size).contiguous()

    # decode
    hypos = [Hypothesis([self.vocab.SOS], [], decoder_hidden, [], [], 1)]
    results= [] 
    backup_results = []
    step = 0
    while hypos and step < 2 * max_out_len and len(results) < beam_size:  
      
      latest_tokens = torch.tensor([h.tokens[0] for h in hypos], device=DEVICE)
      decoder_hidden = torch.cat([h.dec_hidden for h in hypos], 1)
      
      states = torch.cat([torch.cat(h.dec_states, 0) for h in hypos], 1)
      
      enc_attn_weights = [torch.cat([h.enc_attn_weights[i] for h in hypos], 1)
                            for i in range(step)]
      
      coverage_vector, _ = torch.max(torch.cat(enc_attn_weights), dim=0)  
      
      # run the decoder over the assembled batch to get new info
      decoder_embedded = self.embedding(self.filter_oov(latest_tokens, ext_vocab_size))
      decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = \
        self.decoder(decoder_embedded, decoder_hidden, encoder_outputs,
                     states, coverage_vector,
                     encoder_word_idx=input_tensor, ext_vocab_size=ext_vocab_size)
      top_v, top_i = decoder_output.data.topk(beam_size) 
      # create new hypotheses
      n_hypos = 1 if step == 0 else len(hypos)
      all_hypos = []
      for i in range(n_hypos):
        for j in range(beam_size):
          new_tok = top_i[i][j].item()
          new_prob = top_v[i][j].item()
          if len_in_words:
            non_word = not self.vocab.is_word(new_tok)
          else:
            non_word = new_tok == self.vocab.EOS  # only SOS & EOS don't count
          new_hypo = hypos[i].create_next(new_tok, new_prob,
                                               decoder_hidden[0][i].unsqueeze(0).unsqueeze(0),
                                               self.dec_attn,
                                               dec_enc_attn[i].unsqueeze(0).unsqueeze(0),non_word)
          all_hypos.append(new_hypo)
      # process the new hypotheses
      hypos = []
      complete_results, incomplete_results = []
      for nh in sorted(all_hypos, key=lambda h: -h.avg_log_prob):
        if nh.tokens[-1] == self.vocab.EOS:  
          # complete hypotheses
          # If this hypothesis is sufficiently long, put in results. Otherwise discard.
          if len(complete_results) < beam_size and min_out_len <= len(nh) <= max_out_len:
            complete_results.append(nh)
        elif len(hypos) < beam_size and len(nh) < max_out_len:  
          # incomplete hypotheses
          hypos.append(nh)
        
      if complete_results:
        results.extend(complete_results)
      
      step += 1
    if len(results) == 0:  
      results = hypos 

    #returns a list of hypotheses sorted by descending average log probability
    return sorted(results, key=lambda h: h.avg_log_prob, reverse=True)[:beam_size]

"""*Code to decode the summaries*"""

def decode_batch_output(decoded_tokens, vocab, oov_dict):
  ind2word = {value:key for key, value in vocab.word2index.items()}
  
  decoded_batch = []
  if not isinstance(decoded_tokens, list):
    decoded_tokens = decoded_tokens.transpose(0, 1).tolist()
  for i, doc in enumerate(decoded_tokens):
    print("doc", doc)
    decoded_doc = []
    for word_idx in doc:
      if word_idx >= len(vocab):
        word = oov_dict.index2word.get((i, word_idx), '<UNK>')
      else:
        word = vocab[word_idx]
      decoded_doc.append(word)
      if word_idx == vocab.EOS:
        break 
        #Stop at end of sentence
    decoded_batch.append(decoded_doc)
  print("\n decoded batch", decoded_batch)

  decoded_batch_out = []
  for doc in decoded_tokens:
    for tok in doc:
      if tok in ind2word.keys():
        print("the word for index ", tok, " is ", ind2word[tok])
        decoded_batch_out.append(ind2word[tok])

  print(decoded_batch_out)
  return decoded_batch_out


def decode_batch(batch, model, vocab, criterion=None, *, pack_seq=True,
                 show_cover_loss=False):
  #Test the `model` on the `batch`, return the decoded textual tokens and the Seq2SeqOutput.
  if not pack_seq:
    input_lengths = None
  else:
    input_lengths = batch["inp_lengths"]
  with torch.no_grad():
    input_tensor = batch["input"].to(DEVICE)
    target_tensor = batch["target"].to(DEVICE)
    out = model(input_tensor, target_tensor, input_lengths, criterion,
                ext_vocab_size=batch.ext_vocab_size, include_cover_loss=show_cover_loss)
    decoded_batch = decode_batch_output(out.decoded_tokens, vocab, batch["oov_dict"])
  target_length = batch["target"].size(0)
  out.loss_value /= target_length
  return decoded_batch, out

def get_rouge_score_cust(preds, actuals, rouge_type):
    rouge = Rouge()
    tot_scores = {'r': 0, 'p': 0, 'f': 0}
    for i, ac in enumerate(actuals):
        scores = rouge.get_scores(preds[i], ac, avg=True)
        for key in scores[rouge_type].keys():
            tot_scores[key] += scores[rouge_type][key]
    for key in tot_scores.keys():
        tot_scores[key] = tot_scores[key]/len(actuals)
    return tot_scores

def rouge(target, *preds):
  scores = []
  for i, targ_eval in enumerate(target):
    scores.append(get_rouge_score_cust(targ_eval, preds[0][i], "rouge-1"))
    scores.append(get_rouge_score_cust(targ_eval, preds[1][i], "rouge-1"))
  return scores

def eval_batch_output(tgt_tensor, vocab, oov_dict, *pred_tensors):
  decoded_batch = [decode_batch_output(pred_tensor, vocab, oov_dict)
                   for pred_tensor in pred_tensors]
  if isinstance(tgt_tensor, torch.Tensor):
    gold_summaries = decode_batch_output(tgt_tensor, vocab, oov_dict)
  else:
    gold_summaries = tgt_tensor
  scores = rouge(gold_summaries, *decoded_batch)
  return scores

def eval_bs_batch(batch, model, vocab, *, pack_seq=True, beam_size=4,
                  min_out_len=1, max_out_len=None,):
  assert len(batch["examples"]) == 1
  with torch.no_grad():
    input_tensor = batch["input"].to(DEVICE)
    hypotheses = model.beam_search(input_tensor, batch["inp_lengths"],
                                   batch["oov_dict"].ext_vocab_size, beam_size, min_out_len=min_out_len,
                                   max_out_len=max_out_len, len_in_words=False)
  to_decode = [h.tokens for h in hypotheses]
  
  decoded_batch = decode_batch_output(to_decode, vocab, batch["oov_dict"])
  print("\n decoded", decoded_batch)
  print(batch["examples"][0][1])
  print(batch["examples"][0][0])
  file_content = [format_nicely(decoded_batch[0]), format_nicely(batch["examples"][0][1]), format_tokens(batch["examples"][0][0])]
  return file_content


def eval_bs(test_set, vocab, model):
  test_gen = test_set.generator(1, vocab, None, True)
  n_samples = int(len(test_set.pairs))

  model.eval()
  prog_bar = tqdm(range(1, n_samples + 1))

  df = pd.DataFrame(columns = ['predicted', 'episode_description', 'transcript'])

  for i in prog_bar:
    batch = next(test_gen)
    file_content = eval_bs_batch(batch, model, vocab, pack_seq=True,
                                         beam_size=4,
                                         min_out_len=60,
                                         max_out_len=140
                                         )
    if file_content:
      file_content = pd.DataFrame(file_content)
      vertical_concat = pd.concat([df, file_content], axis=0)
  vertical_concat.to_csv("/content/drive/MyDrive/Dissertation/data/pg_results.csv")

"""*Format the scores to write*"""

def format_nicely(tokens, newline= '<P>', for_rouge=False):
  tokens = filter(lambda t: t not in {'<PAD>', '<SOS>', '<EOS>', '<UNK>'}, tokens)
  tokens = list(tokens)
  tokens_sorted = ''.join(map(str, tokens))
  return tokens_sorted

"""*Main code to train and generate summaries*"""

if __name__ == "__main__":
  filename = "/content/drive/MyDrive/Dissertation/data/pickled_for_colab.csv"
  dataset = Dataset(filename)
  vocab = dataset.build_vocab()
  model = Seq2Seq(vocab)
  train_gen = dataset.generator(1, vocab, vocab, True)
  dataset = Dataset("/content/drive/MyDrive/Dissertation/data/pickled_for_colab.csv")
  eval_bs(dataset, vocab, model)

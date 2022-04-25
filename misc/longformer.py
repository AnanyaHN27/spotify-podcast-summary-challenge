# -*- coding: utf-8 -*-
"""longformer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1alFNfhbTJOX6-LNSkBEi9CwXSdzeSVTX
"""

!pip install transformers
!pip install nlp
!pip install datasets
!pip install rouge_score rouge_score

"""*Imports*"""

import nlp
import logging
from transformers import LongformerTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch

"""*Setting configurations, loading data*"""

logging.basicConfig(level=logging.INFO)

model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
model.to("cuda")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# load train, validation and test data
train_dataset = load_dataset('csv', data_files='/content/drive/MyDrive/Dissertation/data/pickled_for_colab.csv', split="train[:75%]")
val_dataset = load_dataset('csv', data_files='/content/drive/MyDrive/Dissertation/data/pickled_for_colab.csv', split="train[:25%]")
test_dataset = load_dataset('csv', data_files='/content/drive/MyDrive/Dissertation/data/pickled_for_colab.csv', split='test')
# load rouge for validation
rouge = nlp.load_metric("rouge", experiment_id=2)

# enable gradient checkpointing for longformer encoder
model.encoder.config.gradient_checkpointing = True

# decoding
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4
encoder_length = 4096
decoder_length = 128
batch_size = 16

"""*Decoding and training by instantiating the Trainer*"""

# map data correctly
def map_to_encoder_decoder_inputs(batch):
    inputs = tokenizer(batch["transcript"], padding="max_length", truncation=True, max_length=encoder_length)
    outputs = tokenizer(batch["episode_description"], padding="max_length", truncation=True, max_length=decoder_length)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["global_attention_mask"] = [[1 if i < 128 else 0 for i in range(sequence_length)] for sequence_length in len(inputs.input_ids) * [encoder_length]]
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    # mask loss for padding
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]
    batch["decoder_attention_mask"] = outputs.attention_mask
    return batch

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.eos_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge1 = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid
    rouge2 = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    rougel = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougel"])["rougel"].mid

    return {
        "rouge1_fmeasure": round(rouge1.fmeasure, 2),
        "rouge2_fmeasure": round(rouge2.fmeasure, 2),
        "rougef_fmeasure": round(rougel.fmeasure, 2),
    }

# make train dataset ready
train_dataset = train_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size)

train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "global_attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],)

# same for validation dataset
val_dataset = val_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size)

val_dataset.set_format(
    type="torch", columns=["input_ids", "global_attention_mask", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],)

# set training arguments
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_from_generate=True,
    evaluate_during_training=True,
    do_train=True,
    do_eval=True,
    logging_steps=100,
    save_steps=100,
    eval_steps=100,
    overwrite_output_dir=True,
    warmup_steps=200,
    save_total_limit=5,
    fp16=True,
)

# instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# start training
trainer.train()

"""*Generate summaries*"""

def generate(batch):
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, :decoder_length] = 1
    outputs = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred"] = output_str
    return batch

results = test_dataset.map(generate, batched=True, batch_size=batch_size)

pred_str = results["pred"]
label_str = results["episode_description"]

pd.DataFrame(results).to_csv('/content/drive/MyDrive/Dissertation/data/longformer_results.csv')

#!/usr/bin/env python3
import os, argparse
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

def load_data(src, labels):
    df_txt = pd.read_csv(src, sep='\t', names=['tweet_id','text'], dtype=str)
    df_lbl = pd.read_csv(labels, sep=':', names=['label','tweet_id'], dtype=str)
    return df_txt.merge(df_lbl, on='tweet_id')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src',    required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--model',  default='bert-base-uncased')
    p.add_argument('--out',    default='bert_preds.npy')
    args = p.parse_args()

    df = load_data(args.src, args.labels)
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    os.makedirs('models/bert_final', exist_ok=True)
    df_ids = df['tweet_id'].tolist()

    ds = Dataset.from_pandas(df[['tweet_id','text','label_id']])
    ds = ds.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def tokenize(batch):
        return tokenizer(batch['text'],
                         padding='max_length',
                         truncation=True,
                         max_length=128)
    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column('label_id', 'labels')
    ds.set_format(type='torch',
                  columns=['input_ids','attention_mask','labels','tweet_id'])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(le.classes_)
    )

    metric = evaluate.load('f1')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {'f1': metric.compute(
            predictions=preds, references=labels, average='macro'
        )['f1']}

    training_args = TrainingArguments(
        output_dir='bert_out',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy='epoch',
        save_strategy='no',
        logging_steps=50,
        learning_rate=2e-5
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    preds_output = trainer.predict(ds['test'])
    logits = preds_output.predictions
    probs  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    labels = np.array(ds['test']['labels'])
    tweet_ids = np.array(ds['test']['tweet_id'])

    np.save(args.out, {
        'tweet_ids': tweet_ids,
        'labels':    labels,
        'probs':     probs
    })
    print(f"Saved BERT test‑set probs to {args.out}")

    trainer.save_model('models/bert_final')
    tokenizer.save_pretrained('models/bert_final')
    print("Saved fine‑tuned BERT ➞ models/bert_final")

if __name__ == '__main__':
    main()

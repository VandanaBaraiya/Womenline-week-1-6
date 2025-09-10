# Minimal example for fine-tuning a summarization model on a CSV dataset with columns: text,summary
# Usage:
# python training/finetune.py --train_csv /path/train.csv --eval_csv /path/val.csv --model_name facebook/bart-large-cnn --output_dir ./checkpoints/bart-journal
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import evaluate

def load_dataset(train_csv, eval_csv):
    train_df = pd.read_csv(train_csv)
    eval_df = pd.read_csv(eval_csv)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(eval_df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--eval_csv', required=True)
    parser.add_argument('--model_name', default='facebook/bart-large-cnn')
    parser.add_argument('--output_dir', default='./checkpoints/journal-summarizer')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(examples):
        model_inputs = tokenizer(examples['text'], max_length=512, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['summary'], max_length=128, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    train_ds, eval_ds = load_dataset(args.train_csv, args.eval_csv)
    train_ds = train_ds.map(preprocess, batched=True)
    eval_ds = eval_ds.map(preprocess, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: round(v, 4) for k, v in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False,
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to: {args.output_dir}")

if __name__ == "__main__":
    main()

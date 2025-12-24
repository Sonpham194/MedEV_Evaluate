import argparse
import numpy as np

from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import sacrebleu
import torch
import transformers as hf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="VietAI/envit5-translation",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="medev_en2vi",
        help="Directory of saved DatasetDict (with train/validation/test)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_debug_envit5_en2vi",
        help="Where to save checkpoints / logs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use small subset of data for quick CPU test",
    )
    return parser.parse_args()


def load_dataset_small(data_dir: str, debug: bool) -> DatasetDict:
    ds = load_from_disk(data_dir)

    if not debug:
        return ds

    def subsplit(dset, n):
        n = min(len(dset), n)
        return dset.select(range(n))

    print("DEBUG MODE: using small subset of the dataset")
    small = DatasetDict(
        {
            "train": subsplit(ds["train"], 50),
            "validation": subsplit(ds["validation"], 10),
            "test": subsplit(ds["test"], 10),
        }
    )
    for split in small:
        print(f"{split}: {len(small[split])} examples")
    return small


def main():
    args = get_args()

    print("=" * 80)
    print("Using transformers version:", hf.__version__)
    print("=" * 80)

    datasets = load_dataset_small(args.data_dir, debug=args.debug)
    train_ds = datasets["train"]
    eval_ds = datasets["validation"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    max_source_length = 64
    max_target_length = 64

    # NOTE: assumes each example has fields "source" and "target"
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )

        # Target side
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_eval = eval_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    bleu_metric = sacrebleu.metrics.BLEU()

    def postprocess_text(preds, labels):
        # strip spaces; keep labels as a flat list of strings
        preds = [p.strip() for p in preds]
        labels = [l.strip() for l in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        # Support both tuple and EvalPrediction objects
        if hasattr(eval_preds, "predictions"):
            preds, labels = eval_preds.predictions, eval_preds.label_ids
        else:
            preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.array(preds)
        labels = np.array(labels)

        # Guard against negative token ids in predictions / labels
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else 0
        )
        preds = np.where(preds < 0, pad_id, preds)
        labels = np.where(labels < 0, pad_id, labels)

        # Decode
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels
        )

        # sacrebleu.metrics.BLEU.corpus_score expects:
        #   hypotheses: List[str]
        #   references: List[List[str]]  (one list per reference set)
        bleu = bleu_metric.corpus_score(decoded_preds, [decoded_labels])

        result = {"bleu": bleu.score}

        # Also report average generated length
        prediction_lens = [np.count_nonzero(p != pad_id) for p in preds]
        result["gen_len"] = float(np.mean(prediction_lens))
        return result

    print("CUDA available:", torch.cuda.is_available())

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,              # 1 epoch is enough for debug
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        eval_strategy="steps",           # <- new name; OK in 4.57.3 
        eval_steps=50,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=10,
        predict_with_generate=True,
        # IMPORTANT: for debug we don't set generation_max_length
        # to avoid extra edge-case bugs on CPU
        fp16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Debug training run finished!")


if __name__ == "__main__":
    main()

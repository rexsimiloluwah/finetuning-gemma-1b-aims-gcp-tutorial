import os 
from datasets import load_dataset
from transformers import AutoTokenizer


def load_dolly_dataset(cfg):
    """
    Load Dolly-15k from Hugging Face, GCS, or local disk.
    """
    if cfg.data.source == "hf":
        return load_dataset(cfg.data.hf_path, split="train")

    if cfg.data.source == "local":
        return load_dataset("json", data_files=cfg.data.local_path, split="train")
    
    if cfg.data.source == "gcs":
        gcs_uri = f"gs://{cfg.data.bucket_name}/{cfg.data.gcs_path}"
        # Copy from GCS to a local temp file then load
        local_path = "data/raw/dolly.jsonl"
        os.makedirs("data/raw", exist_ok=True)
        os.system(f"gsutil cp {gcs_uri} {local_path}")
        return load_dataset("json", data_files=local_path, split="train")

    raise ValueError(f"Unsupported data source: {cfg.data.source}")

    raise ValueError(f"Unsupported data source: {cfg.data.source}")


def format_chat_example(example, tokenizer, max_length):
    """
    Convert a Dolly example into Gemma chat format and tokenize it.
    """
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]

    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
        return_dict=True,
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].copy(),
    }


def prepare_training_dataset(cfg):
    """
    Load and tokenize the training dataset for Gemma chat fine-tuning.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    dataset = load_dolly_dataset(cfg)

    if hasattr(cfg.data, "max_train_samples") and cfg.data.max_train_samples:
        dataset = dataset.select(range(cfg.data.max_train_samples))

    dataset = dataset.map(
        lambda x: format_chat_example(x, tokenizer, cfg.model.max_length),
        remove_columns=dataset.column_names,
    )

    return dataset, tokenizer


def prepare_eval_dataset(cfg):
    """
    Load and tokenize the evaluation dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    dataset = load_dataset("json", data_files=cfg.data.eval_path, split="train")

    dataset = dataset.map(
        lambda x: format_chat_example(x, tokenizer, cfg.model.max_length),
        remove_columns=dataset.column_names,
    )

    return dataset
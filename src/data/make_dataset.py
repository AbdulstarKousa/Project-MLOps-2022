# -*- coding: utf-8 -*-
import logging

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from src import _PATH_DATA


def load_data():
    train_datasets = Dataset.from_pandas(
        pd.read_pickle(_PATH_DATA + "/processed/train.pkl")
    )
    test_datasets = Dataset.from_pandas(
        pd.read_pickle(_PATH_DATA + "/processed/test.pkl")
    )
    return train_datasets, test_datasets


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def raw_data(cache=True):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    # Downloads the raw data from Huggingface datasets
    if cache:
        raw_datasets = load_dataset("imdb", cache_dir=_PATH_DATA + "/raw/")
    else:
        # For testing purposes, no cache is used to make tests
        # independent from using the cache
        raw_datasets = load_dataset("imdb")

    return raw_datasets


def process_data(raw_datasets):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    full_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )  #  a small subset of the training data
    full_test_dataset = (
        tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    )  #  a small subset of the testing data

    full_train_dataset.to_pandas().to_pickle(_PATH_DATA + "/processed/train.pkl")
    full_test_dataset.to_pandas().to_pickle(_PATH_DATA + "/processed/test.pkl")


def main():
    raw_datasets = raw_data()
    process_data(raw_datasets)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

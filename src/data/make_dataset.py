# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def load_data():
    train_datasets = Dataset.from_pandas(pd.read_pickle("../../data/processed/train.pkl"))
    test_datasets = Dataset.from_pandas(pd.read_pickle("../../data/processed/test.pkl"))
    return train_datasets, test_datasets


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    os.makedirs(input_filepath, exist_ok=True)
    raw_datasets = load_dataset("imdb",cache_dir=input_filepath)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    full_train_dataset = tokenized_datasets["train"]
    full_test_dataset = tokenized_datasets["test"]

    os.makedirs(output_filepath, exist_ok=True)
    full_train_dataset.to_pandas().to_pickle(output_filepath+"/train.pkl")
    full_test_dataset.to_pandas().to_pickle(output_filepath+"/test.pkl")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

import os
import hydra
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import load_metric
from src.data.make_dataset import load_data
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainingArguments


logger = logging.getLogger(__name__)


# @hydra.main(config_path="../conf", config_name="config.yaml")
def train():#cfg: DictConfig):
    # logger.info((f"Configuration: \n {OmegaConf.to_yaml(cfg)}"))

    logger.info("Load data")
    train_dataset,test_dataset = load_data()

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2, cache_dir="../../models")
    training_args = TrainingArguments("test_trainer")
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)
    trainer.train()

def main():
    """Runs training loop"""
    logger.info("train model")
    train()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
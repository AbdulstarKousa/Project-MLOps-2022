import logging
from datetime import datetime

import hydra
import numpy as np
import torch
from datasets import load_metric
from omegaconf import DictConfig, OmegaConf
from transformers import (AutoModelForSequenceClassification, Trainer,
                          TrainingArguments)

import wandb
from src import _PATH_MODELS
from src.data.make_dataset import load_data

wandb.init(project="BERT", entity="mlops-2022")
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


@hydra.main(config_path="../conf", config_name="config.yaml")
def train(cfg: DictConfig):
    logger.info((f"Configuration: \n {OmegaConf.to_yaml(cfg)}"))
    logger.info("Load data")
    wandb.log(
        {
            "learning_rate": cfg.training.learning_rate,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
        }
    )
    torch.cuda.empty_cache()
    train_dataset, test_dataset = load_data()

    # Loads pretrained BERT model from hugging-face
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, cache_dir=_PATH_MODELS
    )
    date = datetime.today().strftime("%d-%m-%y:%H%M")
    training_args = TrainingArguments(
        output_dir=_PATH_MODELS + "/{}".format(date),
        learning_rate=cfg.training.learning_rate,
        do_train=cfg.training.do_train,
        do_eval=cfg.training.do_eval,
        num_train_epochs=cfg.training.epochs,
        overwrite_output_dir=cfg.training.overwrite_output_dir,
        evaluation_strategy=cfg.training.evaluation_strategy,
        metric_for_best_model=cfg.training.metric_for_best_model,
        per_device_train_batch_size=cfg.training.batch_size,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        save_strategy=cfg.training.save_strategy,
        report_to="wandb",
        dataloader_num_workers=4,  # or 4 add the number of loader
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def main():
    """Runs training loop"""
    logger.info("train model")
    train()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

import logging
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainingArguments

from src import _PATH_MODELS
from src.data.make_dataset import load_data

wandb.init(project="BERT", entity="mlops-2022")
logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config.yaml")
def train(cfg: DictConfig):
    logger.info((f"Configuration: \n {OmegaConf.to_yaml(cfg)}"))
    logger.info("Load data")
    train_dataset,test_dataset = load_data()

    # Loads pretrained BERT model from hugging-face
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2, cache_dir=_PATH_MODELS)
    
    training_args = TrainingArguments("test_trainer", report_to="wandb")
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)
    trainer.train()

def main():
    """Runs training loop"""
    logger.info("train model")
    train()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
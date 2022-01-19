import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from src import _PATH_MODELS
from src.data.make_dataset import load_data

label_dict = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}

model = AutoModelForSequenceClassification.from_pretrained(
    _PATH_MODELS + "/best_model", torchscript=True
)
tokenizer = AutoTokenizer.from_pretrained(_PATH_MODELS + "/best_model")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# classifier.save_pretrained(_PATH_MODELS+"/best_model")

while True:
    query = input()
    result = classifier(query)[0]
    print(label_dict[result["label"]] + " score: " + str(result["score"]))

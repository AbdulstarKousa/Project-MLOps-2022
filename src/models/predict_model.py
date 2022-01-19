from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src import _PROJECT_ROOT

label_dict = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}

model = AutoModelForSequenceClassification.from_pretrained(
    _PROJECT_ROOT + "/best_model", local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(_PROJECT_ROOT + "/best_model")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print("To stop write quit or exit or ctrl-c")
while True:
    query = input()
    if query == "quit" or query == "exit":
        break
    result = classifier(query)[0]
    print(label_dict[result["label"]] + " score: " + str(result["score"]))

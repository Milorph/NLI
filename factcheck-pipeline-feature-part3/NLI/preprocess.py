from datasets import load_dataset
from collections import Counter
from transformers import AutoTokenizer

mnli = load_dataset("nyu-mll/multi_nli")
anli = load_dataset("facebook/anli")
snli = load_dataset("stanfordnlp/snli")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

single_row = mnli["train"][0]
out = tokenizer(single_row["premise"], single_row["hypothesis"], truncation=True)

print("TEXT: ", mnli["train"][0]["premise"])
print(out)
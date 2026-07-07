from datasets import load_dataset

mnli = load_dataset("nyu-mll/multi_nli")
anli = load_dataset("facebook/anli")
snli = load_dataset("stanfordnlp/snli")

print(mnli)
print(mnli["train"][0])  #get a single row
print(mnli["validation_matched"][0])  
print(mnli["validation_mismatched"][0]) 

print("-------")

print(anli)
print(anli["train_r1"][0])
print(anli["dev_r1"][0])
print(anli["test_r1"][0])

print("-------")

print(snli)
print(snli["train"][0])
print(snli["validation"][0])
print(snli["test"][0])

#keep relevant features + check for any useless data

keep_MNLI = ["premise", "hypothesis", "label", "genre"]
mnli = mnli.select_columns(keep_MNLI)


# label distribution per split
for split in mnli:
    print(split, Counter(mnli[split]["label"]))

# count empty/missing text
for split in mnli:
    ds = mnli[split]
    bad = sum(
        1 for ex in ds
        if not ex["premise"] or not ex["hypothesis"]
        or not ex["premise"].strip() or not ex["hypothesis"].strip()
    )
    print(split, "empty premise/hypothesis:", bad)


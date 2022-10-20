import csv
from emoji import demojize
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('melll-uff/bertweetbr')
tokenizer = AutoTokenizer.from_pretrained('melll-uff/bertweetbr')

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

inputs = []
with open("base_globo_treino.csv") as csvfile:
  csvreader = csv.reader(csvfile, delimiter =';')
  next(csvreader)
  c = 0
  for row in csvreader:
    inputs.append(row[2])
    c = c + 1
    if c>10:
      break

tokenizer.demojizer = lambda x: demojize(x, language='pt')
encoded_inputs = [tokenizer.normalizeTweet(s) for s in inputs]

for s in encoded_inputs:
  print(s)
  print(classifier(s))
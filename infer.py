import torch
import json
import torch.nn as nn
from bpetok import bpe,build_vocab,encode,decode
from datapip import TextDataset
from model import TinyGPT

with open("input.txt", "r") as f:
    corpus = f.read()

with open("tokenizer.json", "r") as f:
    rules = json.load(f)

vocab = build_vocab(rules)
encoded = encode(corpus, rules, vocab)

model = TinyGPT(256,8,128,len(vocab),4)
check = torch.load("checkpoint_epoch150.pt")
model.load_state_dict(check["model_state_dict"])

prompt = input("Enter your prompt: ")
context = torch.tensor([encode(prompt,rules,vocab)])

model.eval()
times = 500
temp = 0.8
for _ in range(times):
    context = context[:, -128:]
    gen = model(context)
    gen = gen[:, -1, :]
    probs = nn.functional.softmax(gen/temp, dim=-1)
    next_token = torch.multinomial(probs, 1)
    context = torch.cat([context, next_token], dim=1)
    
print(decode(context[0].tolist()))
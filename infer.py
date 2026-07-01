import torch
import json
import torch.nn as nn
import tiktoken
from datapip import TextDataset
from model import TinyGPT

with open("input.txt", "r") as f:
    corpus = f.read()

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

model = TinyGPT(256,8,128,vocab_size,4)
check = torch.load("checkpoint_epoch150.pt")
model.load_state_dict(check["model_state_dict"])

prompt = input("Enter your prompt: ")
context = torch.tensor([enc.encode(prompt)])

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
    
print(enc.decode(context[0].tolist()))
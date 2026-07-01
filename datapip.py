import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import urllib.request
from bpetok import bpe, build_vocab, encode

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,data,block_size):
        super().__init__()
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, i):
        src  = torch.tensor(self.data[i:i+self.block_size])
        target = torch.tensor(self.data[i+1:i+self.block_size+1])
        return (src,target)

with open("input.txt", "r") as f:
    corpus = f.read()

print(len(corpus))

rules = bpe(corpus, n=100)
vocab = build_vocab(rules)
encoded = encode(corpus, rules, vocab)
print(len(encoded))

block_size = 128
dataset = TextDataset(encoded, block_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

x, y = next(iter(dataloader))
print(x.shape, y.shape)
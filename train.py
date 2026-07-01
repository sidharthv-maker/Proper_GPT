import torch
import tiktoken
import torch.nn as nn
from model import TinyGPT
from datapip import TextDataset
from bpetok import bpe, build_vocab, encode
from torch.utils.data import DataLoader
import wandb
#wandb is used for tracking and plotting the training runs, useful for checking how the run is going

with open("input.txt", "r") as f:
    corpus = f.read()

enc = tiktoken.get_encoding("gpt2")
encoded = enc.encode(corpus)
vocab_size = enc.n_vocab

dataset = TextDataset(encoded,128)
dset = DataLoader(dataset, batch_size=128, shuffle=True)
model = TinyGPT(256, 8, 128, vocab_size, 4)
#All the scaler related lines are only needed to run on colab, not needed otherwise
scaler = torch.amp.GradScaler('cuda')
optim = torch.optim.Adam(params=model.parameters(), lr=0.001)
lossfn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optim,
    max_lr= 0.001,
    total_steps= 150*len(dset),
    pct_start=0.05
)
accum = 4

wandb.init(project="proper-gpt", config={
    "emb_dim": 256,
    "head_num": 8,
    "seq_len": 128,
    "trans_num": 4,
    "batch_size": 128,
    "lr": 0.001,
    "epochs": 150,
    "bpe_merges": 100
})

for epoch in range(150):
    total_loss = 0
    i = 0
    for x,y in dset:
        i += 1
        with torch.amp.autocast('cuda'):
            output = model(x)
            loss = lossfn(output.view(-1,vocab_size), y.view(-1))
            loss = loss/accum
        scaler.scale(loss).backward()
        if (i+1)%accum == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            optim.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/150  Loss: {total_loss/len(dset):.4f}")
    wandb.log({"loss": total_loss / len(dset), "epoch": epoch})
    #save progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': total_loss / len(dset),
        }, f"checkpoint_epoch{epoch+1}.pt")
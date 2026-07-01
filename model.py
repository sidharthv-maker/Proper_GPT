import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MHSelfAttention(nn.Module):
    def __init__(self, emb_dim, head_num, seq_len, dropout = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.head_dim = emb_dim//head_num
        self.WQ = nn.Linear(emb_dim, emb_dim)
        self.WK = nn.Linear(emb_dim, emb_dim)
        self.WV = nn.Linear(emb_dim, emb_dim)
        self.WO = nn.Linear(emb_dim, emb_dim)

        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        Q = Q.view(B, T, self.head_num, self.head_dim).transpose(1,2)
        K = K.view(B, T, self.head_num, self.head_dim).transpose(1,2)
        V = V.view(B, T, self.head_num, self.head_dim).transpose(1,2)
        score = Q @ K.transpose(-2,-1)
        score = score/(self.emb_dim ** 0.5)
        score = score.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(score, -1)
        weights = self.drop(weights)
        
        out = weights @ V
        out = out.transpose(1,2).contiguous()
        out = out.view(B, T, self.emb_dim)
        out = self.WO(out)
        return out

class Transformer(nn.Module):
    def __init__(self, emb_dim, head_num, seq_len, dropout = 0.1):
        super().__init__()
        self.attn = MHSelfAttention(emb_dim, head_num, seq_len, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.layer = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*emb_dim, emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
       out = self.attn(x)
       x = x + out
       x = self.norm1(x)
       out = self.layer(x)
       x = x+out
       x = self.norm2(x)
       return x

class TinyGPT(nn.Module):
    def __init__(self, emb_dim, head_num, seq_len, vocab_size, trans_num, dropout = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.seq_len = seq_len
        self.head_dim = emb_dim//head_num

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.posemb = nn.Embedding(seq_len, emb_dim)
        self.mod = nn.ModuleList([Transformer(emb_dim, head_num, seq_len, dropout) for _ in range(trans_num)])
        self.norm = nn.LayerNorm(emb_dim)
        self.lin = nn.Linear(emb_dim, vocab_size)
        self.apply(self.weights)

    def forward(self, x):
        x = x[:, :self.seq_len]
        tok = self.emb(x)
        _ , T, _ = tok.shape
        pos = self.posemb(torch.arange(T))
        x = tok + pos  
        for block in self.mod:
            x = block(x)
        x = self.norm(x)
        out = self.lin(x)
        return out

    def weights(self,module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
    

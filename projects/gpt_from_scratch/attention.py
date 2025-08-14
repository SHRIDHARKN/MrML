import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, num_heads, d_model, seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.wq = nn.Linear(self.d_model, self.d_model)
        self.wk = nn.Linear(self.d_model, self.d_model) 
        self.wv = nn.Linear(self.d_model, self.d_model)
        self.ffn = nn.Linear(self.d_model, self.d_model)
        self.d_k = self.d_model // self.num_heads
        self.seq_len = seq_len
        
    def forward(self, x):
        
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = q.view(-1, self.seq_len, self.num_heads, self.d_k)
        v = v.view(-1, self.seq_len, self.num_heads, self.d_k)
        k = k.view(-1, self.seq_len, self.num_heads, self.d_k)
        attention_wts = torch.matmul(q, k.transpose(-2, -1))
        attention_wts = attention_wts / self.d_model**0.5
        attention_scores = torch.softmax(attention_wts, dim=-1)
        x = torch.matmul(attention_scores, v)
        x = x.view(-1, self.seq_len, self.d_model)
        x = self.ffn(x)
        return x
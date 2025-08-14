from tokenizer import tokenizer
from positional_encoding import PositionalEncoding
from attention import AttentionHead
import torch.nn as nn
import torch

class GPT(nn.Module):
    def __init__(self, tokenizer, d_model=64, max_len=256, num_heads=4):
        super(GPT, self).__init__()
        self.random = torch.manual_seed(0)
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.max_len = max_len
        self.positional_encoder = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)
        self.embedding_layer = nn.Embedding(self.tokenizer.n_vocab, self.d_model)
        self.pad_token_id = self.tokenizer.encode("####")
        self.num_heads = num_heads
        self.attention_head = AttentionHead(
                                        num_heads=self.num_heads, 
                                        d_model=self.d_model, 
                                        seq_len=self.max_len
                                        )
        self.num_decoders = 2
        self.decoder_blocks = nn.ModuleList([
                                            AttentionHead(
                                                num_heads=self.num_heads, 
                                                d_model=self.d_model, 
                                                seq_len=self.max_len
                                            ) for _ in range(self.num_decoders)
                                        ])

    def forward(self, x):
        x = self.tokenizer.encode_batch(x)
        x = [t+self.pad_token_id*(self.max_len-len(t)) for t in x]
        print(x)
        # x = x + self.pad_token_id * (self.max_len - len(x))
        # x = torch.tensor(x).unsqueeze(0)
        x = torch.tensor(x)
        x = self.embedding_layer(x)
        x = self.positional_encoder(x)
        # x = self.decoder_blocks[0](x)
        for block in self.decoder_blocks:
            x = block(x)
        return x
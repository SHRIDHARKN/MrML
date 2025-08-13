import torch
import math 
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # create zero tensor
        pe = torch.zeros(max_len, d_model)
        
        # create tensor representing position
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 10000 ^ ab = e ^ ( ab * ln (10000) )
        log_val = math.log(10000.0) # 10000
        pos_val = torch.arange(0, d_model, 2).float()/d_model # 2i / d part 
        div_term = log_val * pos_val
            
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add an extra dimension to `pe` so it can be added to a batch of tensors
        pe = pe.unsqueeze(0)
        
        # Register the positional encoding tensor as a buffer. This means it's part
        # of the model's state but is not a trainable parameter.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: This is your input tensor, which holds the token embeddings. 
        # Its shape is (batch_size, sequence_length, d_model)
        # x.size(1): This method returns the size of the tensor along the first 
        # dimension (index 1). In this case, that's the sequence_length. So, if you 
        # have a sequence of 10 tokens, x.size(1) will be 10.
        x = x + self.pe[:, :x.size(1), :]
        return x
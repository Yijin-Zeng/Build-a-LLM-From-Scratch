import torch
import torch.nn as nn

class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias):
        # context_length: maximum possible sequence length
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_Q = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_K = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.output_proj = nn.Linear(d_out, d_out)

        # move the mask into the appropriate device (CPU or GPU) along with out model
        mask = torch.tril(torch.ones(context_length, context_length))
        # 'mask' is the name we will use to access it later as self.mask
        self.register_buffer('mask', mask)

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        queries = self.W_Q(x)
        keys = self.W_K(x)
        values = self.W_V(x)

        # reshape into [batch_size, num_tokens, num_heads, head_dim]
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # transpose so the shape becomes: [batch_size, num_heads, num_tokens, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2) 
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(-2, -1) # the first dimension is batch size
        mask = self.mask[:num_tokens, :num_tokens] # num_tokens <= context_length
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf')) # apply the mask
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        # shape of context_vec: [batch_size, num_heads, num_tokens, head_dim]
        context_vec = attn_weights @ values

        # transform the shape back into [batch_size, num_tokens, num_heads, head_dim]
        context_vec = context_vec.transpose(1,2)

        # concatenate along the num_heads dimension
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)

        # linear projection to distribute the outputs of each head 
        context_vec = self.output_proj(context_vec)

        return  context_vec
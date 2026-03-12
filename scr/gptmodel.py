import torch
import torch.nn as nn
from attentionLayers import MultiHeadMaskedAttention

class GPTModel(nn.Module):
    def __init__(self, cfg):
        # cfg: configurations
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias = False)

    def forward(self, in_idx):
        # pass the batch
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device = in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

class LayerNorm(nn.Module):
    # implement layernorm
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        var = x.var(dim= -1, keepdim=True,  unbiased=True) # biased or unbiased estimation
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

class GELU(nn.Module):
    # implement GELU activation function
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))))
    

class FeedForward(nn.Module):
    # implement a feedforward layer with GELU with interal expand
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']),
                                    GELU(),
                                    nn.Linear(4*cfg['emb_dim'], cfg['emb_dim']))
    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadMaskedAttention(
            d_in = cfg['emb_dim'], 
            d_out = cfg['emb_dim'], 
            context_length = cfg['context_length'], 
            dropout = cfg['drop_rate'], 
            num_heads = cfg['n_heads'], 
            qkv_bias = cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):

        # attention layer
        short_cut = x
        s = self.norm1(x)
        x = self.att(s)
        x = self.drop_shortcut(x)
        x = x + short_cut # skip connection

        # feedforward layer
        short_cut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + short_cut

        return x
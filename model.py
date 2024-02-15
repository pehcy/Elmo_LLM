import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
from einops_exts import rearrange_many
import math

from typing import Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RotaryPositionalEmbedding(nn.Module):
    '''
    Initializes rotary positional embeddings class.

    Parameters
    ----------
    embed_dim:      int
            the embedding dimension of transformers model.

    Returns
    -------
    tensor
    '''
    def __init__(self, embed_dim, base=10_000) -> None:
        super(RotaryPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.base = base

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, '... (j d) -> ... j d', j=2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x):
        seq_len, device = x.shape[1], x.device
        theta = 1. / (self.base ** (torch.arange(0, self.embed_dim, 2).float() / self.embed_dim)).to(device)
        seq_idx = torch.arange(seq_len, device=x.device).float()
        freqs = torch.einsum("i,j->ij", seq_idx, theta)
        
        positions = torch.cat([freqs, freqs], dim=-1)
        return (x * positions.cos()) + (self._rotate_half(x) * positions.sin())


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_len: int, dropout: Optional[float] = 0.1) -> None:
        '''
        Initializes sinusoidal positional encoding class.

        Parameters
        ----------
        embed_dim:      int
            the embedding dimension of self-attention
        max_seq_len:    int
            the maximum number of words in a string
        dropout:        float
            the probability of dropping out data

        Returns
        -------
        pe: torch.Tensor
        '''
        super(SinusoidalPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(max_seq_len, embed_dim)
        pe = torch.zeros(max_seq_len, embed_dim)

        pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        wavelen = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        
        angle = pos * wavelen

        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)

        # store embedding data into persistent buffer
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = self.embed(x) * math.sqrt(self.embed_dim)
        pe = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        if x.is_cuda:
            pe = pe.cuda()
        
        x = x + pe
        return self.dropout(x)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class RMSNorm(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        p:Optional[float] = -1,
        eps:float = 1e-6, 
        bias:Optional[bool] = False
    ) -> None:
        '''
        Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.

        Reference: https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
        '''
        super(RMSNorm, self).__init__()
        self.embed_dim = embed_dim
        self.p = p
        self.alpha = nn.Parameter(torch.ones(embed_dim))
        self.register_parameter('alpha', self.alpha)
        self.bias = bias
        self.eps = eps

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(embed_dim))
            self.register_parameter('offset', self.offset)

    def forward(self, x: torch.Tensor):
        if self.p < 0 or self.p > 1:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.embed_dim
        else:
            partial_size = int(self.embed_dim * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.embed_dim - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        
        rms_x = norm_x * d_x ** (-0.5)
        x_normalized = x / (rms_x + self.eps)

        if self.bias:
            return self.alpha * x_normalized + self.offset

        return self.alpha * x_normalized


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout: Optional[float] = 0.1, debug_mode: bool = False) -> None:
        '''
        Initializes multihead attention class

        Parameters
        ----------
        d_model:    int
            the embedding dimension of self-attention
        n_head:     int
            the number of heads used in multihead attention

        Returns
        -------
        None
        '''
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.to_q = nn.Linear(d_model, self.d_k * n_heads)
        self.to_k = nn.Linear(d_model, self.d_k * n_heads)
        self.to_v = nn.Linear(d_model, self.d_k * n_heads)

        self.q_rope = RotaryPositionalEmbedding(d_model)
        self.k_rope = RotaryPositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.d_k * n_heads, d_model)

    def forward(self, q, k, v, mask=None):
        # (batch size, max seq len, 512)
        max_seq_len = q.shape[1]
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = self.q_rope(q)
        k = self.k_rope(k)

        # q = self.q_rope(q, max_seq_len)
        # k = self.k_rope(k, max_seq_len)

        q, k, v = rearrange_many((q, k, v), 'b c (h w) -> b w c h', h=self.d_k)

        scores = torch.matmul(q, k.permute(0,1,3,2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = torch.matmul(weights, v)

        output = rearrange(context, "b c h w -> b h (c w)")
        output = self.out(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim=2048, dropout=0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            SwiGLU(),
            nn.Dropout(p=dropout),
            nn.Linear(ff_dim // 2, embed_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.layernorm = RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.multihead = MultiheadAttention(d_model=embed_dim, n_heads=n_heads, dropout=dropout)
        self.ffwd = FeedForward(embed_dim=embed_dim, dropout=dropout)

    def forward(self, embeddings, mask):
        '''
        Parameters
        ----------
        vec_seq:    tuple[int, int, int]
            vector sequence of shape (batch size, sequence length, embedding dim)
        mask:       Tensor
            the source mask of encoder, a.k.a mask over input sequence
            with shape (batch size, 1, sequence length)

        Returns
        -------
        Tensor of shape (batch size, sequence length, embedding dim)
        '''
        x = self.dropout(self.multihead(embeddings, embeddings, embeddings, mask))
        x = self.layernorm(x + embeddings)
        ffwd_out = self.dropout(self.ffwd(x))
        encoded = self.layernorm(ffwd_out + x)
        return encoded


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout: Optional[float] = 0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.layernorm = RMSNorm(embed_dim)
        self.mha1 = MultiheadAttention(d_model=embed_dim, n_heads=n_heads)
        self.mha2 = MultiheadAttention(d_model=embed_dim, n_heads=n_heads)
        self.ffwd = FeedForward(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, embeddings, encoded, src_mask, target_mask):
        query = self.mha1(embeddings, embeddings, embeddings, target_mask)
        # query = self.dropout(query)
        query = self.layernorm(query + embeddings)
        interacted = self.dropout(self.mha2(query, encoded, encoded, src_mask))
        interacted = self.layernorm(interacted + query)
        ffwd_out = self.dropout(self.ffwd(interacted))
        decoded = self.layernorm(ffwd_out + interacted)
        return decoded


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, word_map) -> None:
        super(Transformer, self).__init__()
        
        self.vocab_size = len(word_map)
        self.pe = SinusoidalPositionalEncoding(embed_dim=d_model, max_seq_len=self.vocab_size)

        self.embeddings = nn.Embedding(self.vocab_size, d_model)

        self.encoders = nn.ModuleList([
            EncoderLayer(embed_dim=d_model, n_heads=n_heads) for _ in range(n_layers)])
        
        self.decoders = nn.ModuleList([
                DecoderLayer(d_model, n_heads) for _ in range(n_layers)])
        
        self.logits = nn.Linear(d_model, self.vocab_size)
    
    def encode_stack(self, src_words, src_mask):
        # src_embeddings = self.pe(src_words)
        src_embeddings = self.embeddings(src_words)

        # fold left for each encoder layer
        # [acc := layer(acc) for layer in self.encoder]
        for layer in self.encoders:
            src_embeddings = layer(src_embeddings, src_mask)
        
        return src_embeddings
    
    def decode_stack(self, target_words, target_mask, src_embeddings, src_mask):
        # target_embeddings = self.pe(target_words)
        target_embeddings = self.embeddings(target_words)

        # fold left for each encoder layer
        # [acc := layer(acc) for layer in self.encoder]
        for layer in self.decoders:
            target_embeddings = layer(
                embeddings=target_embeddings, 
                encoded=src_embeddings,
                src_mask=src_mask,
                target_mask=target_mask
            )
        
        return target_embeddings

    def forward(self, src_words, src_mask, target_words, target_mask):
        encoded = self.encode_stack(src_words, src_mask)
        decoded = self.decode_stack(target_words, target_mask, encoded, src_mask)
        out = F.log_softmax(self.logits(decoded), dim=2)
        return out
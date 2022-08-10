import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
from d2l import torch as d2l
from options import cfg

# def scaled_dot_product(q, k, v, mask=None):
#     d_k = q.size()[-1]
#     attn_logits = torch.matmul(q, k.transpose(-2, -1))
#     attn_logits = attn_logits / math.sqrt(d_k)
#     if mask is not None:
#         attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
#     attention = F.softmax(attn_logits, dim=-1)
#     values = torch.matmul(attention, v)
#     return values, attention

# class MultiheadAttention(nn.Module):

#     def __init__(self, input_dim, embed_dim, num_heads):
#         super().__init__()
#         assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         # Stack all weight matrices 1...h together for efficiency
#         # Note that in many implementations you see "bias=False" which is optional
#         self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
#         self.o_proj = nn.Linear(embed_dim, embed_dim)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         # Original Transformer initialization, see PyTorch documentation
#         nn.init.xavier_uniform_(self.qkv_proj.weight)
#         self.qkv_proj.bias.data.fill_(0)
#         nn.init.xavier_uniform_(self.o_proj.weight)
#         self.o_proj.bias.data.fill_(0)

#     def forward(self, x, mask=None, return_attention=False):
#         batch_size, seq_length, embed_dim = x.size()
#         qkv = self.qkv_proj(x)

#         # Separate Q, K, V from linear output
#         qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
#         qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
#         q, k, v = qkv.chunk(3, dim=-1)

#         # Determine value outputs
#         values, attention = scaled_dot_product(q, k, v, mask=mask)
#         values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
#         values = values.reshape(batch_size, seq_length, embed_dim)
#         o = self.o_proj(values)

#         if return_attention:
#             return o, attention
#         else:
#             return o

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MM_MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MM_MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Q，K，V [batch_size, len_seq, len_paris, d_model]
        # valid_lens [batch_size,] or [batch_size, no. of queries]
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
		# QKV [batch_size*num_head, len_sqe, d_model/num_head]
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)
 
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)



        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
from d2l import torch as d2l
from utils import cfg, cfg_from_file
from omegaconf import DictConfig, OmegaConf

cfg_from_file('./config/birds_train.yml')

def transpose_qkv(X, num_heads):
    print("X shape: ", X.shape[0], X.shape[1])
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
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MM_MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens = 1024, num_heads = 8, bias=False):
        super(MM_MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens

        # 强制 hid_dim 必须整除 h
        assert num_hiddens % num_heads == 0
        self.W_q = nn.Linear(1024, self.num_hiddens, bias=bias)
        self.W_k = nn.Linear(1024, self.num_hiddens, bias=bias)
        self.W_v = nn.Linear(1024, self.num_hiddens, bias=bias)
        self.fc = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.do = nn.Dropout(cfg.MM_ATT.dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([num_hiddens // num_heads]))

    def forward(self, query, key, mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        bsz = query.shape[0]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(key)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.num_heads, self.num_hiddens //
                   self.num_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.num_heads, self.num_hiddens //
                   self.num_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.num_heads, self.num_hiddens //
                   self.num_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / 8 #8 | 16 #/ self.scale

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
            
        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.num_heads * (self.num_hiddens // self.num_heads))
        x = self.fc(x)
        
        x = x.reshape(bsz, -1)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from options import cfg

print(cfg.CNNRNN_RNN.input_size)
class multi_attention(nn.Module):
    def __init__(self, in_size, hidden_size, n_heads):
        super(multi_attention, self).__init__()
        self.att_heads = nn.ModuleList()
        for x in range(n_heads):
            self.att_heads.append(attention(in_size, hidden_size))
    def forward(self, input):
        out, self.alpha = [], []
        for head in self.att_heads:
            o = head(input)
            out.append(o) 
            # save the attention matrices to be able to use them in a loss function
            self.alpha.append(head.alpha)
        # return the resulting embedding 
        return torch.cat(out, 1)

class attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, in_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, input):
        # calculate the attention weights
        self.alpha = self.softmax(self.out(nn.functional.tanh(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.alpha * input, 1)
        # return the resulting embedding
        return x 

class CNN_RNN_ENCODER(nn.Module):
    def __init__(self):
        super(CNN_RNN_ENCODER,self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=cfg.CNNRNN.in_channels,out_channels=cfg.CNNRNN.hid_channels,
                              kernel_size=cfg.CNNRNN.kernel_size,stride=cfg.CNNRNN.stride,
                              padding=cfg.CNNRNN.padding)
        self.Conv2 = nn.Conv1d(in_channels=cfg.CNNRNN.hid_channels,out_channels=cfg.CNNRNN.out_channels,
                              kernel_size=cfg.CNNRNN.kernel_size,stride=cfg.CNNRNN.stride,
                              padding=cfg.CNNRNN.padding)

        self.bnorm1 = nn.BatchNorm1d(cfg.CNNRNN.hid_channels)
        self.bnorm2 = nn.BatchNorm1d(cfg.CNNRNN.out_channels)
        if cfg.CNNRNN.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(cfg.CNNRNN_RNN.input_size, cfg.CNNRNN_RNN.hidden_size , cfg.CNNRNN_RNN.num_layers, batch_first=True, dropout=cfg.CNNRNN_RNN.dropout,
                          bidirectional=cfg.CNNRNN_RNN.bidirectional)
        elif cfg.CNNRNN.rnn_type == 'GRU':
            self.rnn = nn.GRU(cfg.CNNRNN_RNN.input_size, cfg.CNNRNN_RNN.hidden_size , cfg.CNNRNN_RNN.num_layers, batch_first=True, dropout=cfg.CNNRNN_RNN.dropout,
                          bidirectional=cfg.CNNRNN_RNN.bidirectional)
        else:
            raise NotImplementedError

        self.att = multi_attention(in_size = cfg.CNNRNN_ATT.in_size, hidden_size = cfg.CNNRNN_ATT.hidden_size, n_heads = cfg.CNNRNN_ATT.n_heads)

    def forward(self, input, l):
        input = input.transpose(2,1)
        x = self.Conv1(input)
        x = self.bnorm1(x)
        x = self.Conv2(x)
        x = self.bnorm2(x)

        # update the lengths to compensate for the convolution subsampling
        l = [int((y-(self.Conv1.kernel_size[0]-self.Conv1.stride[0]))/self.Conv1.stride[0]) for y in l]
        l = [int((y-(self.Conv2.kernel_size[0]-self.Conv2.stride[0]))/self.Conv2.stride[0]) for y in l]
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(2,1), l, batch_first=True)
        # self.rnn.flatten_parameters()
        x, hx = self.rnn(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)

        if cfg.SPEECH.self_att:
            x = self.att(x)
        else:
            x = x.mean(dim=1)
        x = nn.functional.normalize(x, p=2, dim=1)    
        return x

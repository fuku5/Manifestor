import math
import torch
from torch import nn
import torch.nn.functional as F

class Attention_model(nn.Module):
    def __init__(self, n_feature, n_hidden):
        super().__init__()
        self.n_feature = n_feature
        self.layers = nn.Sequential(
                nn.Linear(n_feature, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_feature**2),
                )

    def forward(self, x):
        orig_shape = x.shape
        weights = self.layers(x).reshape(orig_shape[:-1] + (self.n_feature, self.n_feature))
        torch.softmax(weights, axis=len(orig_shape))
        y = (x.unsqueeze(len(orig_shape)-1) * weights).sum(axis=len(orig_shape))
        assert x.shape == y.shape
        return y


class Meta_model_MLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(n_hidden, n_out),
                )

    def forward(self, x):
        return self.layers(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=105):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Meta_Transformer(nn.Module):
    def __init__(self, n_in=8, n_feature=8, n_out=3, n_head=2, n_layers=2, n_hidden=1024, dropout=0.5, encode=True):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        if not encode:
            assert n_in == n_feature
        self.model_type = 'Transformer'
        if encode:
            self.encoder = nn.Linear(n_in, n_feature)
        self.pos_encoder = PositionalEncoding(n_feature, dropout)
        encoder_layers = TransformerEncoderLayer(n_feature, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encode = encode
        self.n_feature = n_feature
        self.decoder = nn.Linear(n_feature, n_out)
        # for CLS token
        self.embedding = nn.Embedding(1, n_feature)

        self.bn = nn.BatchNorm1d(n_feature)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.encode:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask=None):
        # src.shape(sequence_length, batch_size, feature_number)
        if self.encode:
            zeros = torch.zeros(src.shape[1], dtype=torch.long).to(src.device)
            cls = self.embedding(zeros) * math.sqrt(self.n_feature)
            src = self.encoder(src) * math.sqrt(self.n_feature)
            src = torch.cat([cls.unsqueeze(0), src], axis=0)
            if mask is not None:
                falses = torch.zeros(src.shape[1], dtype=torch.bool).to(src.device)
                mask = torch.cat([falses.unsqueeze(1), mask], axis=1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = output[0]
        output = self.bn(output)
        #output = self.norm(output)
        output = self.decoder(output)
        return output

class Guesser_Transformer(nn.Module):
    def __init__(self, n_in=8, dim_utter=3, n_feature=8, n_out=3, n_head=2, n_layers=2, n_hidden=1024, dropout=0.5, encode=True):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        if not encode:
            assert n_in == n_feature
        self.model_type = 'Transformer'
        if encode:
            self.encoder = nn.Linear(n_in, n_feature)
        self.pos_encoder = PositionalEncoding(n_feature*2, dropout, max_len=501)
        encoder_layers = TransformerEncoderLayer(n_feature*2, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encode = encode
        self.n_feature = n_feature
        self.decoder = nn.Linear(n_feature*2, n_out)
        # for CLS token
        self.embedding = nn.Embedding(1, n_feature*2)
        # for embeddding utterance
        self.u_embedding = nn.Embedding(dim_utter, n_feature)

        # trial
        self.bn = nn.BatchNorm1d(n_feature*2)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.encode:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, utterance, mask=None):
        # src.shape(sequence_length, batch_size, feature_number)
        if self.encode:
            zeros = torch.zeros(src.shape[1], dtype=torch.long).to(src.device)
            cls = self.embedding(zeros) * math.sqrt(self.n_feature*2)
            src = self.encoder(src)
            utterance = self.u_embedding(utterance).transpose(1, 0)
            src = torch.cat([src, utterance], axis=2) * math.sqrt(self.n_feature*2)
            src = torch.cat([cls.unsqueeze(0), src], axis=0)
            if mask is not None:
                falses = torch.zeros(src.shape[1], dtype=torch.bool).to(src.device)
                mask = torch.cat([falses.unsqueeze(1), mask], axis=1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = output[0]
        output = self.bn(output)
        output = self.decoder(output)
        return output


if __name__ == '__main__':
    import numpy as np
    meta = Meta_Transformer(n_feature=4, n_out=3)
    x = torch.arange(320).type(torch.float).reshape(20,2,8)
    print(x)
    y = meta(x)
    print(y)
    print(y.shape)


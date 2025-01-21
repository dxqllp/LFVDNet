import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from einops import rearrange, repeat

from Model.Layers import EncoderLayer
from Model.Modules import Attention, ScaledDotProductAttention_bias

PAD = 0


def get_len_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 2
    seq = seq.ne(PAD)
    seq[:, 0] = 1
    return seq.type(torch.float)


def get_attn_key_pad_mask_K(seq_k, seq_q, transpose=False):
    """ For masking out the padding part of key sequence. """
    # [B,L_q,K]
    if transpose:
        seq_q = rearrange(seq_q, 'b l k -> b k l 1')
        seq_k = rearrange(seq_k, 'b l k -> b k 1 l')
    else:
        seq_q = rearrange(seq_q, 'b k l -> b k l 1')
        seq_k = rearrange(seq_k, 'b k l -> b k 1 l')

    return torch.matmul(seq_q, seq_k).eq(PAD)


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    sz_b, len_s, type_num = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)

    subsequent_mask = rearrange(subsequent_mask, 'l l -> b k l l', b=sz_b, k=type_num)
    return subsequent_mask


class FFNN(nn.Module):
    def __init__(self, input_dim, hid_units, output_dim):
        self.hid_units = hid_units
        self.output_dim = output_dim
        super(FFNN, self).__init__()

        self.linear = nn.Linear(input_dim, hid_units)
        self.W = nn.Linear(hid_units, output_dim, bias=False)

    def forward(self, x):
        x = self.linear(x)
        x = self.W(torch.tanh(x))
        return x


class Value_Encoder(nn.Module):
    def __init__(self, hid_units, output_dim, num_type):
        self.hid_units = hid_units
        self.output_dim = output_dim
        self.num_type = num_type
        super(Value_Encoder, self).__init__()

        self.encoder = nn.Linear(1, output_dim)

    def forward(self, x, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        x = rearrange(x, 'b l k -> b l k 1')
        x = self.encoder(x)
        return x * non_pad_mask


class Event_Encoder(nn.Module):
    def __init__(self, d_model, num_types):
        super(Event_Encoder, self).__init__()
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=PAD)

    def forward(self, event):
        # event = event * self.type_matrix
        event_emb = self.event_emb(event.long())
        return event_emb


class Time_Encoder(nn.Module):
    def __init__(self, embed_time, num_types):
        super(Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.k_map = nn.Parameter(torch.ones(1, 1, num_types, embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:  # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')

        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1)  # [B,L,1,D]
        out = torch.mul(out, self.k_map)
        # return out * non_pad_mask # [B,L,K,D]
        return out


class MLP_Tau_Encoder(nn.Module):
    def __init__(self, embed_time, num_types, hid_dim=16):
        super(MLP_Tau_Encoder, self).__init__()
        self.encoder = FFNN(1, hid_dim, embed_time)
        self.k_map = nn.Parameter(torch.ones(1, 1, num_types, embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:  # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')

        # out1 = F.gelu(self.linear1(tt))
        tt = self.encoder(tt)
        tt = torch.mul(tt, self.k_map)
        return tt * non_pad_mask  # [B,L,K,D]


class Selection_window(nn.Module):
    def __init__(self, windowlen, d_model, dropout):
        super().__init__()
        self.windowlen = windowlen
        self.offset = nn.Sequential(nn.Linear(d_model, 2 * d_model),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(2 * d_model, d_model),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_model, 1, bias=False))

    @torch.no_grad()
    def selectwindow(self, S):
        window = torch.arange(0, S * 2, 2, dtype=torch.int64)
        return window

    def forward(self, input, event_time):
        B, K, L, D = input.shape
        index_sample = self.selectwindow(self.windowlen).clone().detach()
        input = rearrange(input, 'b k l d -> (b k) l d')
        input_s = input[:, index_sample, :]  # (b k) l d -> (b k) s d
        offset = self.offset(input_s).to(input.device)
        offset = rearrange(offset, '(b k) s 1 -> b k s 1', b=B, k=K)
        index_sample = index_sample.unsqueeze(0).unsqueeze(0).unsqueeze(-1).float().to(input.device)
        pos = (index_sample + offset).to(input.dtype)

        limit_length = torch.argmax(event_time, -1).to(input.dtype)
        limit_length = rearrange(limit_length, 'b -> b 1 1 1')
        mask = pos <= limit_length
        mask = mask.float()

        _, _, S, _ = pos.shape
        pos = rearrange(pos, 'b k s 1 -> (b k) s 1')
        pos_grid = pos.unsqueeze(1).expand(B * K, 1, S, 2)
        pos_grid = pos_grid / (L - 1) * 2 - 1
        pos_grid[..., 1] = 0.
        input = rearrange(input, 'bk l d -> bk d 1 l')
        pos_grid = pos_grid.contiguous()
        input = input.contiguous()
        # (N,C,Hin,Win) (N,Hout,Wout,2) (N,C,Hout,Wout)
        output = F.grid_sample(input, pos_grid, mode='bilinear', align_corners=True)  # (b k) d 1 s
        output = output.squeeze(2)
        output = rearrange(output, '(b k) d s -> b k s d', b=B, k=K)
        output = output
        mask = mask.squeeze(-1)
        return output, mask


class TAOS_T(nn.Module):

    def __init__(
            self, opt,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):

        super().__init__()
        self.opt = opt
        self.d_model = d_model
        self.embed_time = d_model
        self.median_len = opt.median_len

        # event type embedding
        self.event_enc = Event_Encoder(d_model, num_types)
        self.type_matrix = torch.tensor([int(i) for i in range(1, num_types + 1)]).to(opt.device)
        self.type_matrix = rearrange(self.type_matrix, 'k -> 1 1 k')
        self.num_types = num_types
        self.task = opt.task

        self.value_enc = Value_Encoder(hid_units=d_inner, output_dim=d_model, num_type=num_types)
        self.learn_time_embedding = Time_Encoder(self.embed_time, num_types)
        self.w_t = nn.Linear(1, num_types, bias=False)

        self.tau_encoder = MLP_Tau_Encoder(self.embed_time, num_types)
        self.fc = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.attention = ScaledDotProductAttention_bias(opt.d_model, 1, opt.d_model, opt.d_model,
                                                        temperature=opt.d_model ** 0.5,
                                                        attn_dropout=opt.dropout)

        self.dsam1 = Doubly_Self_Attention_Module(opt, d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.dsam2 = Doubly_Self_Attention_Module(opt, d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.selectdown = Selection_window(self.median_len, self.d_model, dropout)

        self.agg_attention = Attention_Aggregator(d_model, task=opt.task)

    def forward(self, event_time, event_value, non_pad_mask, tau=None, return_almat=False):
        """ Encode event sequences via masked self-attention. """
        '''
        non_pad_mask: [B,L,K]
        slf_attn_mask: [B,K,LQ,LK], the values to be masked are set to True
        len_pad_mask: [B,L], pick the longest length and mask the remains
        '''
        # embedding
        tem_enc_k = self.learn_time_embedding(event_time, non_pad_mask)  # [B,L,1,D], [B,L,K,D]
        tem_enc_k = rearrange(tem_enc_k, 'b l k d -> b k l d')  # [B,K,L,D]

        value_emb = self.value_enc(event_value, non_pad_mask)
        value_emb = rearrange(value_emb, 'b l k d -> b k l d')  # [B,K,L,D]

        self.type_matrix = self.type_matrix.to(non_pad_mask.device)
        # event_emb = self.type_matrix * non_pad_mask
        event_emb = self.type_matrix
        event_emb = self.event_enc(event_emb)
        event_emb = rearrange(event_emb, 'b l k d -> b k l d')  # [B,K,L,D]

        tau_emb = self.tau_encoder(tau, non_pad_mask)
        tau_emb = rearrange(tau_emb, 'b l k d -> b k l d')

        h0 = value_emb + tau_emb + event_emb + tem_enc_k

        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b k l')
        non_pad_mask = rearrange(non_pad_mask, 'b k l -> b k l 1')
        q = tem_enc_k + tau_emb
        k = (tem_enc_k + tau_emb) * non_pad_mask
        v = h0 * non_pad_mask
        non_pad_mask = rearrange(non_pad_mask, 'b k l 1 -> b k 1 l')
        non_pad_mask = non_pad_mask.eq(PAD)
        h0, _ = self.attention(q, k, v, non_pad_mask)
        h0 = self.fc(h0)

        len_pad_mask = get_len_pad_mask(event_time)  # b l
        non_pad_mask = repeat(len_pad_mask, 'b l -> b k l', k=self.num_types)
        h0 = self.dsam1(h0, non_pad_mask)

        h0, mask = self.selectdown(h0, event_time)

        h0 = self.dsam2(h0, mask)

        output = self.agg_attention(h0, rearrange(mask, 'b k s -> b k s 1'))  # [B,D]

        return output


class Doubly_Self_Attention_Module(nn.Module):
    def __init__(self, opt, d_model, d_inner, n_head, d_k, d_v, dropout):
        super(Doubly_Self_Attention_Module, self).__init__()
        self.opt = opt
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, opt=opt)
            for _ in range(opt.n_layers)])

    def forward(self, h0, non_pad_mask=None):
        for enc_layer in self.layer_stack:
            h0, _, _ = enc_layer(h0, non_pad_mask=non_pad_mask)
        return h0


class Pool_Classifier(nn.Module):

    def __init__(self, dim, cls_dim):
        super(Pool_Classifier, self).__init__()
        self.classifier = nn.Linear(dim, cls_dim)

    def forward(self, ENCoutput):
        """
        input: [B,L,K,D]
        """
        b, l, k = ENCoutput.size(0), ENCoutput.size(1), ENCoutput.size(2)
        ENCoutput = rearrange(ENCoutput, 'b l k d -> (b l) d k')
        ENCoutput = F.max_pool1d(ENCoutput, k).squeeze()
        ENCoutput = rearrange(ENCoutput, '(b l) d -> b d l', b=b, l=l)
        ENCoutput = F.max_pool1d(ENCoutput, l).squeeze(-1)
        return self.classifier(ENCoutput)


class Attention_Aggregator(nn.Module):
    def __init__(self, dim, task):
        super(Attention_Aggregator, self).__init__()
        self.task = task
        self.attention_len = Attention(dim * 2, dim)
        self.attention_type = Attention(dim * 2, dim)

    def forward(self, ENCoutput, mask=None):
        """
        input: [B,K,L,D], mask: [B,K,L]
        """
        if self.task == "active":
            mask = rearrange(mask, 'b k l 1 -> b l k 1')
            ENCoutput = rearrange(ENCoutput, 'b k l d -> b l k d')
            ENCoutput, _ = self.attention_type(ENCoutput, mask)  # [B L D]
        else:
            ENCoutput, _ = self.attention_len(ENCoutput, mask)  # [B,K,D]
            ENCoutput, _ = self.attention_type(ENCoutput)  # [B,D]
        return ENCoutput


class Classifier(nn.Module):

    def __init__(self, dim, type_num, cls_dim, activate=None):
        super(Classifier, self).__init__()
        # self.linear1 = nn.Linear(dim, type_num)
        # # self.activate = nn.Sigmoid()
        # self.linear2 = nn.Linear(type_num, cls_dim)
        self.linear = nn.Linear(dim, cls_dim)

    def forward(self, ENCoutput):
        """
        input: [B,L,K,D], mask: [B,L,K]
        """
        # ENCoutput = self.linear1(ENCoutput)
        # # if self.activate:
        # #     ENCoutput = self.activate(ENCoutput)
        # ENCoutput = self.linear2(ENCoutput)
        ENCoutput = self.linear(ENCoutput)
        return ENCoutput

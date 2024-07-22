
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding
import numpy as np
import time

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.hidden_size = configs.d_model  # hidden_size


        # Embedding
        self.intra_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.inter_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.seq_len, kernel_size=(2, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(self.seq_len, self.d_model, kernel_size=(2, 1), stride=(1, 1)),
            nn.ReLU()
        )

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )


        self.projector1 = nn.Linear(configs.pred_len * 2, configs.pred_len, bias=True)
        self.dropout = nn.Dropout(configs.dropout)


        # Decoder
        if self.task_name == 'long_term_forecast' :
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projector = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projector = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

        if self.task_name == 'long_term_forecast' :
            self.out_len = configs.pred_len
        if self.task_name == 'anomaly_detection':
            self.out_len = configs.seq_len
        if self.task_name == 'classification':
            self.out_len = configs.seq_len


    def fc_layers(self, x):
        B, C, T = x.shape
        x = nn.Linear(T, self.out_len, bias=True).to(x.device)(x)
        return x

    def inter_frequency(self, x):
        x_fft = torch.fft.rfft(x, dim=1)
        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag

        # Combining real and imaginary parts into a complex tensor
        x_fft = torch.stack([x_fft_real, x_fft_imag], dim=-1).permute(0, 2, 1, 3)

        y = self.conv_layers(x_fft)
        y = torch.view_as_complex(y)

        x_time = torch.fft.irfft(y, dim=2)

        B, C, T = x_time.shape
        output = nn.Linear(T, self.out_len, bias=True).to(x_time.device)(x_time)

        return output

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape
        # intra
        intra_in = self.intra_embedding(x_enc, x_mark_enc)
        intra_out, attns = self.encoder(intra_in, attn_mask=None)
        intra_out = self.projector(intra_out).permute(0, 2, 1)[:, :, :N]

        # inter
        inter_in = self.inter_embedding(x_enc, x_mark_enc)
        inter_out = self.inter_frequency(inter_in)
        inter_out = inter_out.permute(0, 2, 1)[:, :, :N]

        co_out = self.dropout(inter_out + intra_out)

        # De-Normalization from Non-stationary Transformer
        co_out = co_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        co_out = co_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return co_out


    def anomaly_detection(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape  # B L N

        intra_in = self.intra_embedding(x_enc, x_mark_enc)
        intra_out, attns = self.encoder(intra_in, attn_mask=None)
        intra_out = self.projector(intra_out).permute(0, 2, 1)[:, :, :N]

        inter_in = self.inter_embedding(x_enc, x_mark_enc)
        inter_out = self.inter_frequency(inter_in)
        inter_out = inter_out.permute(0, 2, 1)[:, :, :N]

        co_out = self.dropout(inter_out + intra_out)

        co_out = co_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        co_out = co_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        return co_out

    def classification(self, x_enc, x_mark_enc):

        intra_in = self.intra_embedding(x_enc, None)
        intra_out, attns = self.encoder(intra_in, attn_mask=None)
        intra_out = self.act(intra_out)
        intra_out = self.dropout(intra_out)
        intra_out = intra_out.reshape(intra_out.shape[0], -1)
        intra_out = self.projector(intra_out)


        inter_in = self.inter_embedding(x_enc, None)
        inter_out = self.inter_frequency(inter_in)
        inter_out = self.act(inter_out)
        inter_out = inter_out.reshape(inter_out.shape[0], -1)
        _, L = inter_out.shape
        inter_out = nn.Linear(L, self.num_class).to(inter_out.device)(inter_out)

        co_out = self.dropout(inter_out + intra_out)

        return co_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' :
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

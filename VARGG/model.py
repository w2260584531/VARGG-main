
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import Sequential, BatchNorm, InstanceNorm
from typing import Callable, Iterable, Union, Tuple, Optional
import torchvision.transforms as transforms
import logging
import math
class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_layer = nn.MultiheadAttention(embed_size, num_heads, bias=False)
    def forward(self, x):
        attention_output, _ = self.attention_layer(x, x, x)
        return attention_output

class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, stddev=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev + self.mean
            return x + noise
        return x

class VARGG_model(nn.Module):
    def __init__(self, 
                input_dim, 
                Conv_type = 'GatedGraphConv',
                linear_encoder_hidden = [64,40],
                linear_decoder_hidden = [64],
                conv_hidden = [40,32,16,8],
                p_drop = 0.01,
                dec_cluster_n = 15,
                alpha = 1.0,
                activate="relu",
                ):
        super(VARGG_model, self).__init__()
        self.input_dim = input_dim
        self.Conv_type = Conv_type
        self.alpha = alpha
        self.conv_hidden = conv_hidden
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n
        self.head=4
        current_encoder_dim = self.input_dim
        self.attention=nn.Sequential()
        self.attention.add_module(f'attention',SelfAttention(input_dim,8))

        self.gaussian_noise = GaussianNoise(mean=0.0, stddev=0.1)

        self.encoder = nn.Sequential()
        for le in range(len(linear_encoder_hidden)):
            self.encoder.add_module(f'encoder_L{le}', 
                                    buildNetwork(current_encoder_dim, self.linear_encoder_hidden[le], self.activate, self.p_drop))
            if le==0:
                self.encoder.add_module(f'attention_L',SelfAttention(self.linear_encoder_hidden[le],4))
            current_encoder_dim = linear_encoder_hidden[le]
        current_decoder_dim = linear_encoder_hidden[-1] + conv_hidden[-1]

        self.decoder = nn.Sequential()
        for ld in range(len(linear_decoder_hidden)):
            self.decoder.add_module(f'decoder_L{ld}',
                                    buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate, self.p_drop))
            self.encoder.add_module(f'attention_L',SelfAttention(self.linear_decoder_hidden[ld],8)) 
            current_decoder_dim= self.linear_decoder_hidden[ld]
        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}', nn.Linear(self.linear_decoder_hidden[-1], 
                                self.input_dim))



          


        self.Conv_type == "ResGatedGraphConv"
        from torch_geometric.nn import ResGatedGraphConv
        self.conv = Sequential('x, edge_index', [
                        (ResGatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ELU(),                        
                        (ResGatedGraphConv(conv_hidden[0]* 2, conv_hidden[1]* 2), 'x1, edge_index -> x2'),
                        BatchNorm(conv_hidden[1]* 2),
                        nn.ELU(),                        
                        (ResGatedGraphConv(conv_hidden[1]* 2, conv_hidden[2]* 2), 'x2, edge_index -> x3'),
                        BatchNorm(conv_hidden[2]* 2),
                        nn.ELU(),                        
                        (ResGatedGraphConv(conv_hidden[2]* 2, conv_hidden[3]* 2), 'x3, edge_index -> x4'),
                        BatchNorm(conv_hidden[3]* 2),
                        nn.ELU(), 
                        ])
        self.conv_mean = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden[3]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
        self.conv_logvar = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden[3]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])


        self.dc = InnerProductDecoder(p_drop)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.linear_encoder_hidden[-1] + self.conv_hidden[-1]))
        self.cluster_momentum = 0.90  
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def encode(
        self, 
        x, 
        adj,
        ):
        feat_x = self.encoder(x)
        guss_x=self.gaussian_noise(feat_x)
        conv_x=self.conv(guss_x,adj)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj), feat_x
    def reparameterize(
        self, 
        mu, 
        logvar,
        ):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    def target_distribution(
        self, 
        target
        ):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def vargg_loss(
        self, 
        decoded, 
        x, 
        preds, 
        labels, 
        mu, 
        logvar, 
        n_nodes, 
        norm, 
        mask=None, 
        mse_weight=10, 
        bce_kld_weight=0.1,
        ):
        mse_fun = torch.nn.MSELoss()
        mse_loss = mse_fun(decoded, x)

        if mask is not None:
            preds = preds * mask
            labels = labels * mask

        bce_logits_loss = norm * F.cross_entropy(preds, labels)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
              1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return mse_weight * mse_loss + bce_kld_weight* (bce_logits_loss + KLD)
    
    def update_cluster_centers(self, z):

        current_centers = self.cluster_layer.data
        new_centers = (1 - self.cluster_momentum) * current_centers + self.cluster_momentum * z.mean(dim=0)
        self.cluster_layer.data = new_centers

    def forward(
        self, 
        x, 
        adj,
        ):
        x=self.attention(x)
        # x = self.gaussian_noise(x)
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)
        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z
            


def buildNetwork(
    in_features,
    out_features,
    activate="leaky_relu",
    p_drop=0.0
):
    net = []
    net.append(nn.Linear(in_features, out_features))
    nn.init.kaiming_uniform_(net[-1].weight, nonlinearity='relu')
    
    net.append(nn.BatchNorm1d(out_features, momentum=0.1, eps=1e-5))
    
    if activate == "relu":
        net.append(nn.ELU())
    elif activate == "leaky_relu":
        net.append(nn.LeakyReLU(0.01)) 
    elif activate == "prelu":
        net.append(nn.PReLU())
    elif activate == "selu":
        net.append(nn.SELU())
    elif activate == "sigmoid":
        net.append(nn.Sigmoid())
    
    if p_drop > 0:
        net.append(nn.Dropout(p_drop))
    
    return nn.Sequential(*net)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(
        self, 
        dropout, 
        act=torch.sigmoid,
        ):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(
        self, 
        z,
        ):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t())) 
        return adj 
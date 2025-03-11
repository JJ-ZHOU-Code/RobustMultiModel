import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import reparameterize, poe, prior_expert

class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, feats_dim, latent_dim=256, ff_size=1024, num_layers=2, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()
        
        self.feats_dim = feats_dim
                
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        
        self.input_feats = self.feats_dim
        
        self.muQuery = nn.Parameter(torch.randn(self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(self.latent_dim))

        nn.init.normal_(self.muQuery, std=1e-6)
        nn.init.normal_(self.sigmaQuery, std=1e-6)

                        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, x):
        nfeats, bs, feats_dim = x.shape

        xseq = torch.cat((self.muQuery[None]+torch.zeros(1,x.shape[1],x.shape[-1], dtype=x.dtype, device=x.device), self.sigmaQuery[None]+torch.zeros(1,x.shape[1],x.shape[-1], dtype=x.dtype, device=x.device), x), axis=0)

        final = self.seqTransEncoder(xseq)
        mu = final[0]
        logvar = final[1]
            
        return mu, logvar

class Decoder_Specific(nn.Module):
    def __init__(self, ablation=None, use_condition=False, latent_dim=256, input_dim=256, hidden_dim=1024, feature_types=6):
        super().__init__()
        self.use_condition = use_condition
        self.feature_types = feature_types
        self.latent_dim = latent_dim
        self.latent_generators = nn.ModuleList([nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
        ) for _ in range(feature_types)])
        
        decoder_input_dim = latent_dim 
        self.ablation = ablation
        if self.use_condition and self.ablation != "decoder_z":
            decoder_input_dim = latent_dim * 2 

        self.decoders = nn.ModuleList([nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        ) for _ in range(feature_types)])
        
    def forward(self, x, nfeats, condition=None):
        results = {}
        mu_list, logvar_list, z_list, recon_list = [], [], [], []
        for i, decoder in enumerate(self.decoders):
            encoded = self.latent_generators[i](x)
            mu, logvar = torch.chunk(encoded, 2, dim=-1)
            z = reparameterize(mu, logvar)
            if condition is not None and self.ablation != "decoder_z":
                z = torch.cat((z, condition), dim=-1)
            decoded = decoder(z)

            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(z)
            recon_list.append(decoded)

        mu = torch.stack(mu_list, dim=1)
        logvar = torch.stack(logvar_list, dim=1)
        z = torch.stack(z_list, dim=1)
        recon_x = torch.stack(recon_list, dim=0)
        # results['recon_x'] = recon_x
        # results['omic_component_dist'] = (mu, logvar)
        return recon_x, (mu, logvar)

class Decoder_Share(nn.Module):
    def __init__(self, ablation=None, use_condition=False, latent_dim=256, input_dim=256, hidden_dim=1024, feature_types=6):
        super().__init__()
        self.use_condition = use_condition
        self.feature_types = feature_types
        self.latent_dim = latent_dim
        self.latent_generators = nn.ModuleList([nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
        ) for _ in range(feature_types)])
        
        decoder_input_dim = latent_dim 
        self.ablation = ablation
        if self.use_condition and self.ablation != "decoder_z":
            decoder_input_dim = latent_dim * 2 

        self.decoders = nn.ModuleList([nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        ) for _ in range(feature_types)])
        
    def forward(self, x, nfeats, condition=None):
        results = {}
        recon_list = []

        if condition is not None and self.ablation != "decoder_z":
            z = torch.cat((x, condition), dim=-1)

        for i, decoder in enumerate(self.decoders):
            decoded = decoder(z)

            recon_list.append(decoded)

        recon_x = torch.stack(recon_list, dim=0)
        return recon_x, None


class LDVAE(nn.Module):
    def __init__(self, input_dim, use_condition=False, decoder_mode='specific', nfeats=6, hidden_dim=256):
        super(LDVAE, self).__init__()
        self.use_condition = use_condition 
        self.nfeats = nfeats
        self.latent_dim = hidden_dim
        self.encoder = Encoder_TRANSFORMER(feats_dim=input_dim, latent_dim=self.latent_dim, num_layers=1)
        
        # self.decoder = Decoder_TRANSFORMER(feats_dim=input_dim, num_layers=4)
        if decoder_mode == 'specific':
            self.decoder = Decoder_Specific(use_condition=self.use_condition, latent_dim=256, input_dim=256, hidden_dim=512, feature_types=nfeats)
        elif decoder_mode == 'shared':
            self.decoder = Decoder_Share(use_condition=self.use_condition, latent_dim=256, input_dim=256, hidden_dim=512, feature_types=nfeats)
        else:
            raise ValueError('decoder_mode must be either specific or share')

    def experts(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

    def infer(self, x, condition_dist=None):
        mu, logvar = self.encoder(x)
        self.results['omic_dist'] = (mu, logvar)
        if condition_dist is not None:
            bs = x.shape[1]
            mu_joint, logvar_joint = prior_expert((1, bs, self.latent_dim), use_cuda=True)
            mu_cond, logvar_cond = condition_dist
            mu, logvar = torch.cat([mu_joint, mu_cond.unsqueeze(0), mu.unsqueeze(0)], dim=0), \
                    torch.cat([logvar_joint, logvar_cond.unsqueeze(0), logvar.unsqueeze(0)], dim=0)
        
            mu, logvar = self.experts(mu, logvar)
            self.results['joint_dist'] = (mu, logvar)

        return mu, logvar 

    def forward(self, x, condition_dist=None):
        self.results = {}
        x = x.permute((1,0,2))

        if condition_dist is not None:
            mu_cond, logvar_cond = condition_dist
            z_cond = reparameterize(mu_cond, logvar_cond)

        mu, logvar = self.infer(x, condition_dist)
        z = reparameterize(mu, logvar)

        if condition_dist is not None:
            recon_x, omic_component_dist = self.decoder(z, self.nfeats, z_cond)
        else:
            recon_x, omic_component_dist = self.decoder(z, self.nfeats)

        recon_x = recon_x.permute((1,0,2))
        self.results['recon_x'] = recon_x
        self.results['omic_component_dist'] = omic_component_dist
        return self.results
    
    def sample(self, bs=1, condition_dist=None):
        # z = torch.randn(bs, self.latent_dim) # bs, latent_dim
        # z = z.to('cuda')

        if condition_dist is not None: 
            mu, logvar = prior_expert((1, bs, self.latent_dim), use_cuda=True)
            mu_cond, logvar_cond = condition_dist
            mu, logvar = torch.cat([mu, mu_cond.unsqueeze(0)], dim=0), \
                    torch.cat([logvar, logvar_cond.unsqueeze(0)], dim=0)
            mu, logvar = self.experts(mu, logvar)
        else: 
            mu, logvar = prior_expert((bs, self.latent_dim), use_cuda=True)
        
        z_cond = reparameterize(mu, logvar)
        if condition_dist is not None:           
            recon_x, _ = self.decoder(z_cond, self.nfeats, z_cond)
        else:
            recon_x, _ = self.decoder(z_cond, self.nfeats)
        recon_x = recon_x.permute((1,0,2))
        return recon_x

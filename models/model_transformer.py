from cmath import log
import re
from urllib.parse import _ResultMixinStr
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *

from models.nys_transformer import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        mu_token, logvar_token, feat_token = x[:, 0], x[:, 1], x[:, 2:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((mu_token.unsqueeze(1), logvar_token.unsqueeze(1), x), dim=1)
        return x


class VIBTrans(nn.Module):
    def __init__(self, feature_dim=512):
        super(VIBTrans, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.muQuery = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.muQuery, std=1e-6)
        nn.init.normal_(self.sigmaQuery, std=1e-6)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        # nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

        self.fc = nn.Linear(feature_dim, 256)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        muQuery = self.muQuery.expand(B, -1, -1).cuda()
        sigmaQuery = self.sigmaQuery.expand(B, -1, -1).cuda()
        h = torch.cat((muQuery, sigmaQuery, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)

        h = self.fc(h)
        return h[:, 0], h[:, 1], h[:, 2:]

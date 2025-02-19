#! /usr/bin/env python

import torch
import torch.nn.functional as F
from torch import nn
import logging

class SimpleEncClassifier(nn.Module):
    def __init__(self, enc_dims, mlp_dims, dropout=0.2, verbose=1):
        super().__init__()
        self.enc_dims = enc_dims
        self.mlp_dims = mlp_dims
        self.encoder_model = None
        self.mlp_model = None
        self.encoded = None
        self.mlp_out = None
        self.encoder_modules = []
        self.mlp_modules = []
        self.verbose = verbose
        # encoder
        n_stacks = len(self.enc_dims) - 1
        # internal layers in encoder
        for i in range(n_stacks - 1):
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # encoded features layer. no activation.
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))

        # encoder model
        self.encoder_model = nn.Sequential(*(self.encoder_modules))

        # MLP
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            logging.info(f'self.encoder_model')
            logging.info(f'self.mlp_model')
        return

    def update_mlp_head(self, dropout=0.2):
        self.mlp_out = None
        self.mlp_modules = []

        # MLP
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            logging.info(f'{self.encoder_model}')
            logging.info(f'{self.mlp_model}')
        return

    def forward(self, x):
        self.encoded = self.encoder_model(x)
        self.out = self.mlp_model(self.encoded)
        return self.encoded, self.encoded, self.out
    
    def predict_proba(self, x):
        _, _, mlp_out = self.forward(x)
        return mlp_out
    
    def predict(self, x):
        self.encoded = self.encoder_model(x)
        self.out = self.mlp_model(self.encoded)
        preds = self.out.max(1)[1]
        return preds
    
    def encode(self, x):
        self.encoded = self.encoder_model(x)
        return self.encoded
    
class MLPClassifier(nn.Module):
    def __init__(self, mlp_dims, dropout=0.2, verbose=1):
        super().__init__()
        self.mlp_dims = mlp_dims
        self.mlp_model = None
        self.mlp_out = None
        self.mlp_modules = []
        self.verbose = verbose

        # MLP
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            logging.info(f'{self.mlp_model}')
        return

    def forward(self, x):
        self.mlp_out = self.mlp_model(x)
        return self.mlp_out
    
    def predict_proba(self, x):
        mlp_out = self.forward(x)
        return mlp_out
    
    def predict(self, x):
        self.mlp_out = self.mlp_model(x)
        preds = self.mlp_out.max(1)[1]
        return preds
    
    def encode(self, x):
        self.encoded = self.mlp_model[:-2](x)
        return self.encoded

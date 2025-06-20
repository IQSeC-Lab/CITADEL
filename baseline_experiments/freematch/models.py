import torch
import torch.nn.functional as F
from torch import nn
import logging

# === Classifier Definition ===
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

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

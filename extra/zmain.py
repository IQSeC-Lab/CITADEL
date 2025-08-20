# Full Integrated Training Pipeline with Triplet Loss + Centroid Alignment (with Dual Embedding Models)

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datetime import datetime

# === Dataset and Model Definitions ===

class MalwareSiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(MalwareSiameseNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.feature_extractor(x)

class MalwareCentroidEmbedding(nn.Module):
    def __init__(self, input_dim):
        super(MalwareCentroidEmbedding, self).__init__()
        self.transformer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return self.transformer(x)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor - negative, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

class CentroidAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels):
        unique_labels = labels.unique()
        loss = 0.0
        for label in unique_labels:
            class_mask = (labels == label)
            class_embeds = embeddings[class_mask]
            if class_embeds.size(0) == 0:
                continue
            centroid = class_embeds.mean(dim=0)
            loss += ((class_embeds - centroid) ** 2).sum() / class_embeds.size(0)
        return loss / unique_labels.size(0)

class MalwareTripletDataset(Dataset):
    def __init__(self, X, y, num_triplets=5000):
        self.X = X
        self.y = y
        self.num_triplets = num_triplets
        self.triplets = []
        self._create_triplets()

    def _create_triplets(self):
        for _ in range(self.num_triplets):
            idx_anchor = random.randint(0, len(self.y) - 1)
            pos_indices = np.where(self.y == self.y[idx_anchor])[0]
            neg_indices = np.where(self.y != self.y[idx_anchor])[0]
            if len(pos_indices) > 1 and len(neg_indices) > 0:
                idx_pos = random.choice(pos_indices)
                idx_neg = random.choice(neg_indices)
                self.triplets.append((self.X[idx_anchor], self.X[idx_pos], self.X[idx_neg]))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        return torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative)

class MalwareClassifier(nn.Module):
    def __init__(self, embedding_dim=128):
        super(MalwareClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(x)

# === Training Functions ===

def train_triplet_model(model, triplet_loader, optimizer, margin=1.0, epochs=70, device=None):
    model.train()
    triplet_loss_fn = TripletLoss(margin=margin)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x1, x2, x3 in triplet_loader:
            x1, x2, x3 = x1.to(device).float(), x2.to(device).float(), x3.to(device).float()
            v1, v2, v3 = model(x1), model(x2), model(x3)
            loss = triplet_loss_fn(v1, v2, v3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Triplet Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(triplet_loader):.4f}")

def train_centroid_model(ref_model, centroid_model, X_tensor, y_tensor, optimizer, lambda_centroid=1.0, epochs=80, device=None):
    centroid_model.train()
    criterion = CentroidAlignmentLoss()
    for epoch in range(epochs):
        with torch.no_grad():
            ref_embeddings = ref_model(X_tensor.to(device))
        outputs = centroid_model(ref_embeddings.detach())
        loss = criterion(outputs, y_tensor.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Centroid Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def train_classifier(embedding_model, classifier, data_loader, optimizer, criterion, device=None):
    classifier.train()
    for epoch in range(20):
        total_loss = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                emb = embedding_model(x)
            logits = classifier(emb.detach())
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Classifier Epoch [{epoch+1}/20], Loss: {total_loss/len(data_loader):.4f}")

# === Visualization ===
def plot_embeddings(embeddings, labels, title="Embedding Space", save_path="embedding.png"):
    reducer = TSNE(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === Main Experiment ===
def main():
    path = "/home/mhaque3/myDir/data/gen_apigraph_drebin/"
    file_path = f"{path}2012-01to2012-12_selected.npz"
    data = np.load(file_path, allow_pickle=True)
    X, y = data['X_train'], data['y_train']
    y = np.array([0 if label == 0 else 1 for label in y])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    siamese_model = MalwareSiameseNetwork(input_dim=X.shape[1]).to(device)
    centroid_model = MalwareCentroidEmbedding(input_dim=128).to(device)
    classifier = MalwareClassifier().to(device)

    triplet_loader = DataLoader(MalwareTripletDataset(X_scaled, y), batch_size=64, shuffle=True)

    triplet_optimizer = optim.Adam(siamese_model.parameters(), lr=1e-3)
    centroid_optimizer = optim.Adam(centroid_model.parameters(), lr=1e-3)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_triplet_model(siamese_model, triplet_loader, triplet_optimizer, device=device)
    train_centroid_model(siamese_model, centroid_model, X_tensor, y_tensor, centroid_optimizer, device=device)

    # full_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)
    # train_classifier(centroid_model, classifier, full_loader, classifier_optimizer, criterion, device=device)

    with torch.no_grad():
        embeddings = centroid_model(siamese_model(X_tensor.to(device))).cpu().numpy()
        plot_embeddings(embeddings, y, title="Dual Model Embedding", save_path="dual_model_embedding2.png")

if __name__ == "__main__":
    main()


# # Full Integrated Training Pipeline with Triplet Loss + Centroid Alignment

# import os
# import csv
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.manifold import TSNE
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from datetime import datetime

# # === Dataset and Model Definitions ===

# class MalwareSiameseNetwork(nn.Module):
#     def __init__(self, input_dim):
#         super(MalwareSiameseNetwork, self).__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )

#     def forward(self, x):
#         return self.feature_extractor(x)

# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative):
#         pos_dist = torch.norm(anchor - positive, dim=1)
#         neg_dist = torch.norm(anchor - negative, dim=1)
#         loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
#         return loss.mean()

# class CentroidAlignmentLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, embeddings, labels):
#         unique_labels = labels.unique()
#         loss = 0.0
#         for label in unique_labels:
#             class_mask = (labels == label)
#             class_embeds = embeddings[class_mask]
#             if class_embeds.size(0) == 0:
#                 continue
#             centroid = class_embeds.mean(dim=0)
#             loss += ((class_embeds - centroid) ** 2).sum() / class_embeds.size(0)
#         return loss / unique_labels.size(0)

# class MalwareTripletDataset(Dataset):
#     def __init__(self, X, y, num_triplets=5000):
#         self.X = X
#         self.y = y
#         self.num_triplets = num_triplets
#         self.triplets = []
#         self._create_triplets()

#     def _create_triplets(self):
#         for _ in range(self.num_triplets):
#             idx_anchor = random.randint(0, len(self.y) - 1)
#             pos_indices = np.where(self.y == self.y[idx_anchor])[0]
#             neg_indices = np.where(self.y != self.y[idx_anchor])[0]
#             if len(pos_indices) > 1 and len(neg_indices) > 0:
#                 idx_pos = random.choice(pos_indices)
#                 idx_neg = random.choice(neg_indices)
#                 self.triplets.append((self.X[idx_anchor], self.X[idx_pos], self.X[idx_neg]))

#     def __len__(self):
#         return len(self.triplets)

#     def __getitem__(self, idx):
#         anchor, positive, negative = self.triplets[idx]
#         return torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative)

# class MalwareClassifier(nn.Module):
#     def __init__(self, embedding_dim=128):
#         super(MalwareClassifier, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(embedding_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 2)
#         )

#     def forward(self, x):
#         return self.classifier(x)

# # === Training Functions ===

# def train_model_with_centroid_alignment(model, triplet_loader, X_tensor, y_tensor, optimizer, margin=1.0, lambda_centroid=0.5, epochs=70, device=None):
#     model.train()
#     triplet_loss_fn = TripletLoss(margin=margin)
#     centroid_loss_fn = CentroidAlignmentLoss()

#     for epoch in range(epochs):
#         epoch_triplet_loss = 0.0
#         epoch_centroid_loss = 0.0

#         for x1, x2, x3 in triplet_loader:
#             x1, x2, x3 = x1.to(device).float(), x2.to(device).float(), x3.to(device).float()
#             v1, v2, v3 = model(x1), model(x2), model(x3)
#             loss_triplet = triplet_loss_fn(v1, v2, v3)
#             optimizer.zero_grad()
#             loss_triplet.backward()
#             optimizer.step()
#             epoch_triplet_loss += loss_triplet.item()

#         model.eval()
        
#         all_embeddings = model(X_tensor.to(device).float())
#         labels_tensor = y_tensor.to(device).long()

#         model.train()
#         optimizer.zero_grad()
#         loss_centroid = centroid_loss_fn(all_embeddings, labels_tensor)
#         (lambda_centroid * loss_centroid).backward()
#         optimizer.step()
#         epoch_centroid_loss += loss_centroid.item()

#         print(f"Epoch [{epoch+1}/{epochs}] | Triplet Loss: {epoch_triplet_loss/len(triplet_loader):.4f} | Centroid Loss: {epoch_centroid_loss:.4f}")


# def train_classifier(model, classifier, train_loader, classifier_optimizer, classifier_criterion, epochs=40, device=None):
#     model.eval()
#     classifier.train()
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         for x, y in train_loader:
#             x, y = x.to(device), y.to(device)
#             embeddings = model(x).detach()
#             logits = classifier(embeddings)
#             loss = classifier_criterion(logits, y)
#             classifier_optimizer.zero_grad()
#             loss.backward()
#             classifier_optimizer.step()
#             epoch_loss += loss.item()
#         print(f"Epoch [{epoch+1}/{epochs}], Classifier Loss: {epoch_loss/len(train_loader):.4f}")


# # === Optional: t-SNE Plot ===
# def plot_embeddings(embeddings, labels, title="Embedding Space", method="tsne", save_path="embedding.png"):
#     if method == "tsne":
#         reducer = TSNE(n_components=2, random_state=42)
#     else:
#         raise ValueError("Only 'tsne' is currently supported")
#     reduced = reducer.fit_transform(embeddings)
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
#     plt.legend(*scatter.legend_elements(), title="Class")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()


# # === Main Function Example ===
# def main():
#     path = "/home/mhaque3/myDir/data/gen_apigraph_drebin/"
#     file_path = f"{path}2012-01to2012-12_selected.npz"
#     data = np.load(file_path, allow_pickle=True)
#     X, y = data['X_train'], data['y_train']
#     y = np.array([0 if label == 0 else 1 for label in y])

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
#     y_tensor = torch.tensor(y, dtype=torch.long)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = MalwareSiameseNetwork(input_dim=X.shape[1]).to(device)
#     classifier = MalwareClassifier().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
#     classifier_criterion = nn.CrossEntropyLoss()

#     triplet_loader = DataLoader(MalwareTripletDataset(X_scaled, y, num_triplets=5000), batch_size=64, shuffle=True)
#     train_model_with_centroid_alignment(model, triplet_loader, X_tensor, y_tensor, optimizer, device=device)

#     classifier_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)
#     train_classifier(model, classifier, classifier_loader, classifier_optimizer, classifier_criterion, device=device)

#     with torch.no_grad():
#         embeddings = model(X_tensor.to(device)).cpu().numpy()
#         plot_embeddings(embeddings, y, title="Centroid-Aligned Embedding", save_path="embedding_centroid_aligned.png")

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from collections import Counter

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import random

import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler
import csv
import os
from datetime import datetime

from similarity_score import get_high_similar_samples

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MalwareSiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(MalwareSiameseNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # Embedding output
        )

    def forward(self, x):
        return self.feature_extractor(x)  # Returns 128-dimensional embeddings

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor - negative, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()



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
                idx_pos = pos_indices[random.randint(0, len(pos_indices) - 1)]
                idx_neg = neg_indices[random.randint(0, len(neg_indices) - 1)]
                self.triplets.append((self.X[idx_anchor], self.X[idx_pos], self.X[idx_neg]))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        return torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the classifier
class MalwareClassifier(nn.Module):
    def __init__(self, embedding_dim=128):
        super(MalwareClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification (Benign vs Malware)
        )

    def forward(self, x):
        return self.classifier(x)  # Returns logits

# Define classifier training function
def train_classifier(model, classifier, train_loader, classifier_optimizer, classifier_criterion, epochs=40, device=None):
    model.eval()  # Keep contrastive model fixed
    classifier.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x, y = batch  # Features and labels
            x, y = x.to(device), y.to(device)

            # Forward pass
            embeddings = model(x).detach()
            logits = classifier(embeddings)
            loss = classifier_criterion(logits, y)

            # Backward pass
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Classifier Loss: {epoch_loss/len(train_loader):.4f}")

# Define training function
def train_model(model, train_loader, optimizer, loss_function, epochs=70, device=None):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x1, x2, x3 = batch  # Anchor, Positive, Negative for Triplet Loss
            x1, x2, x3 = x1.to(device).type(torch.float32), x2.to(device).type(torch.float32), x3.to(device).type(torch.float32) # Cast to float32

            # Forward pass
            v1, v2, v3 = model(x1), model(x2), model(x3)

            loss = loss_function(v1, v2, v3)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")


# Function to plot embeddings using t-SNE or PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_embeddings(embeddings, labels, title="Embedding Space", method="tsne", save_path="embedding.png"):
    # embeddings = embeddings.detach().cpu().numpy()
    # labels = labels.cpu().numpy()

    # Reduce dimensionality
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be either 'tsne' or 'pca'")
    
    reduced = reducer.fit_transform(embeddings)

    # Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def main():
    # Load malware dataset file
    path = "/home/mhaque3/myDir/data/gen_apigraph_drebin/"

    # Extract features and labels
    file_path = f"{path}2012-01to2012-12_selected.npz"  # Change this to your actual dataset path
    data = np.load(file_path, allow_pickle=True)
    X, y = data['X_train'], data['y_train']
    y = [0 if label == 0 else 1 for label in y]
    # Extract features and labels
    # X = df.iloc[:, 1:-1].values  # Features (skip sha256 and label columns)
    # y = df.iloc[:, -1].values    # Labels (0 = benign, 1 = malware)

    # Normalize features
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # # Convert to PyTorch tensors
    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # y_tensor = torch.tensor(y, dtype=torch.long)

    # Split into Train and Test sets
    # X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    y_train = np.array(y)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y, dtype=torch.long)

    
    
    # Initialize models and optimizers
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:3")
        print(f"🔥 Using device: {device} → {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🔥 Using fallback device: {device}")

    print("✅ Available CUDA devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f" - cuda:{i} → {torch.cuda.get_device_name(i)}")

    strategy = "Sia_triplet_cls_bce_sim_pseudo"

    model = MalwareSiameseNetwork(input_dim=X_train_scaled.shape[1]).to(device)
    classifier = MalwareClassifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.0001, weight_decay=1e-4)

    # Learning rate scheduler for better stability
    model_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    classifier_scheduler = lr_scheduler.StepLR(classifier_optimizer, step_size=5, gamma=0.5)


    classifier_criterion = nn.CrossEntropyLoss()

    # Train the contrastive model on the labeled dataset (2012)
    train_loader = DataLoader(MalwareTripletDataset(X_train_scaled, y_train, num_triplets=5000), batch_size=64, shuffle=True)
    train_model(model, train_loader, optimizer, TripletLoss(), device=device)

    # Generate embeddings
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    # train_embeddings = model(X_train_tensor)

    # Print shape before creating dataset
    print("🔹 Train Shape:", X_train_tensor.shape)  # Should be (N, 128)
    print("🔹 y_train Shape:", y_train.shape)  # Should be (N,)

    # Ensure `y_train` has the correct shape
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).view(-1)  # Ensure it's 1D

    # plot initial embedding space
    with torch.no_grad():
        train_embeddings = model(X_train_tensor.to(device))

        # plot_embeddings(
        #     train_embeddings.detach().cpu().numpy(),
        #     y_train,
        #     title="Initial Embedding Space (2012)",
        #     method="tsne",  # or "pca"
        #     save_path=f"embedding_plots/embedding_2012_tsne.png"
        # )



    # Create dataset
    # assert train_embeddings.shape[0] == y_train_tensor.shape[0], "Size mismatch between embeddings and labels!"
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    classifier_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train classifier on labeled dataset (2012)
    train_classifier(model, classifier, classifier_train_loader, classifier_optimizer, classifier_criterion, device=device)
    train_time = datetime.now().strftime("%m%d_%H%M")
    # Initialize CSV File (Overwrite if Exists)
    csv_filename = f"ssl-malware_performance_{strategy}_{train_time}.csv"
    # Initialize the TSV file (Tab-Separated Values)
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")  # Use tab separator
        writer.writerow(["Year-Month", "Accuracy", "Precision", "Recall", "F1-Score", "TPR", "TNR", "FPR", "FNR"])

    # Active Learning Loop (2013-2018) with model & classifier training
    performance_results = []

    for year in range(2013, 2019):
        for month in range(1, 13):
            print(f"\n🔹 Processing {year}-{month:02d}...")

            # Load new month’s dataset
            file_path = f"{path}{year}-{month:02d}_selected.npz"
            try:
                new_data = np.load(file_path)
            except FileNotFoundError:
                print(f"⚠ No data for {year}-{month:02d}, skipping...")
                continue

            X_new = new_data["X_train"]
            y_true = new_data["y_train"]  # True labels for evaluation
            y_true = (y_true > 0).astype(int)  # Convert to binary labels

            # Normalize new data
            X_new_scaled = scaler.transform(X_new)
            X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

            # **Step 1: Pseudo-Labeling with Cosine Similarity**
            model.eval()
            classifier.eval()
            with torch.no_grad():
                new_embeddings = model(X_new_tensor.to(device))
                # plot_embeddings(
                #     new_embeddings.detach().cpu().numpy(),
                #     y_true,
                #     title=f"Embedding After {year}-{month:02d}",
                #     method="tsne",
                #     save_path=f"embedding_plots/embedding_{year}_{month:02d}_tsne.png"
                # )
                train_embeddings = model(X_train_tensor.to(device))
                pseudo_logits = classifier(new_embeddings)
                pseudo_probs = torch.softmax(pseudo_logits, dim=1)
                pseudo_labels = torch.argmax(pseudo_probs, dim=1)
                indicies, y_pseudo = get_high_similar_samples(
                        train_embeddings.to(device),  # Ensure on the same device
                        y_train_tensor.to(device),    # Ensure on the same device
                        y_train_tensor.to(device),    # Ensure on the same device
                        new_embeddings.to(device),    # Ensure on the same device
                        sm_fn='cosine',
                        num_samples=new_embeddings.shape[0]
                    )


            X_pseudo = X_new_tensor[indicies]
            y_pseudo = torch.tensor(y_pseudo, dtype=torch.long).view(-1)
            # **Step 2: Active Learning - Select High Confidence Pseudo-Labels**
            confidence_threshold = 0.85
            high_confidence_mask = pseudo_probs.max(dim=1).values > confidence_threshold
            # Ensure all tensors are on the same device
            high_confidence_mask = high_confidence_mask.to(X_new_tensor.device)
            pseudo_labels = pseudo_labels.to(X_new_tensor.device)

            # Select high-confidence samples
            X_pseudo = X_pseudo[high_confidence_mask]
            y_pseudo = y_pseudo[high_confidence_mask]

            # **Step 3: Combine Labeled & Pseudo-Labeled Data**
            X_train_tensor = torch.cat([X_train_tensor, X_pseudo])
            y_train_tensor = torch.cat([y_train_tensor, y_pseudo])
            y_train_tensor = y_train_tensor.view(-1) 

            # **Step 4: Retrain Both Models with Updated Data**
            # Train the contrastive model on the labeled dataset (2012)
            new_model_train_loader = DataLoader(MalwareTripletDataset(X_train_tensor, y_train_tensor, num_triplets=5000), batch_size=64, shuffle=True)

            # Retrain contrastive model (Siamese Network)
            train_model(model, new_model_train_loader, optimizer, TripletLoss(), device=device)

            # classifier datasets and loaders
            new_train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            classifier_new_train_loader = DataLoader(new_train_dataset, batch_size=64, shuffle=True)

            # Retrain classifier
            train_classifier(model, classifier, classifier_new_train_loader, classifier_optimizer, classifier_criterion, device=device)

            # Step the learning rate scheduler
            model_scheduler.step()
            classifier_scheduler.step()

            # Print current learning rates
            for param_group in optimizer.param_groups:
                print(f"📉 Model LR: {param_group['lr']}")

            for param_group in classifier_optimizer.param_groups:
                print(f"📉 Classifier LR: {param_group['lr']}")

            print(f"✅ Model & Classifier Updated with {year}-{month:02d} Data")

            # **Step 5: Evaluate Classifier Performance**
            with torch.no_grad():
                test_embeddings = model(X_new_tensor.to(device))
                test_logits = classifier(test_embeddings)
                test_predictions = torch.argmax(test_logits, dim=1).cpu().numpy()

            # Convert true labels to NumPy
            y_true = y_true[:len(test_predictions)]  # Ensure same length
            y_pred = test_predictions
            y_pred = (y_pred > 0).astype(int)

            # Compute Performance Metrics
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Compute Confusion Matrix for TPR, TNR, FPR, FNR
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file, delimiter="\t")  # Use tab separator
                writer.writerow([
                    f"{year}-{month:02d}",
                    f"{acc:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}",
                    f"{tpr:.4f}", f"{tnr:.4f}", f"{fpr:.4f}", f"{fnr:.4f}"
                ])
            print(f"📊 {year}-{month:02d} Performance: Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, TPR={tpr:.4f}, TNR={tnr:.4f}, FPR={fpr:.4f}, FNR={fnr:.4f}")


    # Save final trained models
    torch.save(model.state_dict(), "ssl_model_siamese_{strategy}.pth")
    torch.save(classifier.state_dict(), "ssl_classifier_{strategy}.pth")
    print("🚀 Models trained on 5 years of malware data with active learning!")


if __name__ == "__main__":
    main()



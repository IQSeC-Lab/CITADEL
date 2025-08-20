# Full Integrated Semi-Supervised Active Learning Setup with Improved Siamese Network and Contrastive Loss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Improved Siamese Network
class ImprovedSiameseNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(ImprovedSiameseNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return F.normalize(x, p=2, dim=1)

# Classifier on top of Siamese Embeddings
class MalwareClassifier(nn.Module):
    def __init__(self, embedding_dim=128):
        super(MalwareClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(x)

# Supervised Contrastive Loss
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        loss = -mean_log_prob_pos.mean()
        return loss

# Drift detection using cosine similarity
def cosine_embedding_drift(old_embeddings, new_embeddings):
    cos = nn.CosineSimilarity(dim=1)
    scores = cos(old_embeddings, new_embeddings)
    return 1 - scores.mean().item()

# Memory Buffer class
class MemoryBuffer:
    def __init__(self, max_size=5000):
        self.X = []
        self.y = []
        self.max_size = max_size

    def update(self, X_new, y_new):
        self.X.extend(X_new.cpu())
        self.y.extend(y_new.cpu())
        if len(self.X) > self.max_size:
            self.X = self.X[-self.max_size:]
            self.y = self.y[-self.max_size:]

    def get_tensor_data(self):
        return torch.stack(self.X), torch.tensor(self.y, dtype=torch.long)

# Pseudo-labeling with confidence threshold
def get_high_confidence_pseudo_labels(model, classifier, X_tensor, threshold=0.9):
    with torch.no_grad():
        embeddings = model(X_tensor)
        logits = classifier(embeddings)
        probs = torch.softmax(logits, dim=1)
        pseudo_labels = torch.argmax(probs, dim=1)
        confidence = probs.max(dim=1).values
        mask = confidence > threshold
        return X_tensor[mask], pseudo_labels[mask]

# Model update with memory buffer
def update_model(model, classifier, optimizer, X, y, criterion, device):
    model.train()
    classifier.train()
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True, drop_last=True)
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        embeddings = model(batch_x)
        outputs = classifier(embeddings)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

import csv
from sklearn.metrics import confusion_matrix

# Drift adaptation loop
def concept_drift_adaptation_loop(path, model, classifier, optimizer, criterion, scaler, device):
    csv_filename = "drift_adaptation_performance.tsv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter="	")
        writer.writerow(["Year-Month", "Accuracy", "F1-Score", "Recall", "TPR", "TNR", "FPR", "FNR"])
    buffer = MemoryBuffer(max_size=5000)
    drift_log = []

    for year in range(2013, 2019):
        for month in range(1, 13):
            try:
                data = np.load(f"{path}{year}-{month:02d}_selected.npz")
                X_raw = data["X_train"]
                y_true = (data["y_train"] > 0).astype(int)
            except FileNotFoundError:
                continue

            X_scaled = scaler.transform(X_raw)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

            model.eval()
            classifier.eval()
            with torch.no_grad():
                emb = model(X_tensor)
                out = classifier(emb)
                pred = torch.argmax(out, dim=1).cpu().numpy()

            acc = accuracy_score(y_true, pred)
            f1 = f1_score(y_true, pred)
            recall = recall_score(y_true, pred)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            print(f"🔹 {year}-{month:02d} | Acc: {acc:.4f}, F1: {f1:.4f}, TPR: {tpr:.4f}, TNR: {tnr:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}")

            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file, delimiter="	")
                writer.writerow([
                    f"{year}-{month:02d}", f"{acc:.4f}", f"{f1:.4f}", f"{recall:.4f}",
                    f"{tpr:.4f}", f"{tnr:.4f}", f"{fpr:.4f}", f"{fnr:.4f}"
                ])

            if buffer.X:
                X_mem, _ = buffer.get_tensor_data()
                X_mem = X_mem.to(device)
                old_emb = model(X_mem)
                drift_score = cosine_embedding_drift(old_emb, emb[:old_emb.shape[0]])
                print(f"📈 Drift Score: {drift_score:.4f}")
                drift_log.append((f"{year}-{month:02d}", drift_score))

            X_pseudo, y_pseudo = get_high_confidence_pseudo_labels(model, classifier, X_tensor, threshold=0.9)
            if len(X_pseudo) > 0:
                buffer.update(X_pseudo, y_pseudo)
                X_mem, y_mem = buffer.get_tensor_data()
                update_model(model, classifier, optimizer, X_mem.to(device), y_mem.to(device), criterion, device)
                print(f"✅ Model updated with {len(X_pseudo)} high-confidence samples")

    return drift_log

# The main training loop should be appended below to complete integration
# Removed duplicate main function to unify under concept_drift_main()

    print("🚀 Models trained on 5 years of malware data with active learning!")

def concept_drift_main():
    path = "/home/mhaque3/myDir/data/gen_apigraph_drebin/"
    file_path = f"{path}2012-01to2012-12_selected.npz"
    data = np.load(file_path, allow_pickle=True)
    X, y = data['X_train'], data['y_train']
    y = np.array([0 if label == 0 else 1 for label in y])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ImprovedSiameseNetwork(input_dim=X_tensor.shape[1]).to(device)
    classifier = MalwareClassifier().to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Initial training
    print("🟢 Initial training on 2012 data")
    update_model(model, classifier, optimizer, X_tensor.to(device), y_tensor.to(device), criterion, device)

    # Start the drift adaptation loop
    print("🔁 Starting drift adaptation...")
    drift_log = concept_drift_adaptation_loop(
        path=path,
        model=model,
        classifier=classifier,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        device=device
    )

    # Save model
    torch.save(model.state_dict(), "improved_siamese_drift_adapted.pth")
    torch.save(classifier.state_dict(), "drift_adapted_classifier.pth")
    print("✅ Models saved!")

    # t-SNE plot of embeddings
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        embeddings = model(X_tensor.to(device)).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], c='green', label='Benign', alpha=0.5)
    plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], c='red', label='Malware', alpha=0.5)
    plt.legend()
    plt.title("t-SNE of Malware Embeddings (2012 Data)")
    plt.savefig("tsne_embeddings_2012.png")
    print("📌 t-SNE plot saved as 'tsne_embeddings_2012.png'")

if __name__ == "__main__":
    concept_drift_main()  # Unified entry point with concept drift adaptation

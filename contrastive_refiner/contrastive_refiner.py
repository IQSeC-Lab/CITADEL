import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import pandas as pd

# ---- Model Definitions ----

class ContrastiveAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

class RefinerAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ---- Utility Functions ----

def to_tensor(X):
    return torch.tensor(X, dtype=torch.float32)

def binary_labels(y):
    return (y != 0).astype(np.int64)

def get_centroids(z, y):
    """Compute centroids for each class in latent space."""
    centroids = []
    for label in [0, 1]:
        centroids.append(z[y == label].mean(axis=0))
    return np.stack(centroids)

def uncertainty_scores(logits, method='entropy'):
    """Compute uncertainty scores for a batch of logits."""
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    if method == 'entropy':
        return -np.sum(probs * np.log(probs + 1e-8), axis=1)
    elif method == 'margin':
        part = np.partition(-probs, 1, axis=1)
        return - (part[:, 0] - part[:, 1])
    elif method == 'least_confidence':
        return 1 - np.max(probs, axis=1)
    else:
        raise ValueError("Unknown uncertainty method")

def select_uncertain_samples(scores, top_k=None, threshold=None):
    """Select indices of most uncertain samples."""
    if top_k is not None:
        idx = np.argsort(scores)[-top_k:]
    elif threshold is not None:
        idx = np.where(scores >= threshold)[0]
    else:
        raise ValueError("Specify top_k or threshold")
    return idx

def evaluate_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    roc_auc = roc_auc_score(y_true, y_score)
    # --- Fix for PR-AUC ---
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # Sort recall and precision by recall
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    pr_auc = auc(recall, precision)
    return dict(
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        fnr=fnr, fpr=fpr, roc_auc=roc_auc, pr_auc=pr_auc
    )

# ---- Training Functions ----

import torch.nn.functional as F

def supervised_contrastive_loss(z, y, temperature=0.5):
    """
    z: (batch_size, latent_dim)
    y: (batch_size,) integer labels
    """
    z = F.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T) / temperature  # (B, B)
    labels = y.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)  # (B, B)
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
    mask = mask * logits_mask  # remove self-comparisons

    exp_sim = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss

def train_contrastive_autoencoder(X, y, latent_dim=64, epochs=20, batch_size=128, lr=1e-3, lambda_contrastive=1.0):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = ContrastiveAutoencoder(X.shape[1], latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    recon_criterion = nn.MSELoss()
    dataset = TensorDataset(to_tensor(X), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            z, x_rec = model(xb)
            loss_recon = recon_criterion(x_rec, xb)
            loss_contrast = supervised_contrastive_loss(z, yb)
            loss = loss_recon + lambda_contrastive * loss_contrast
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    return model

def train_refiner_autoencoder(X_uncertain_latent, centroid, latent_dim=64, epochs=20, batch_size=128, lr=1e-3):
    """
    Trains the refiner autoencoder to move uncertain latent samples to the centroid.
    X_uncertain_latent: latent vectors of uncertain samples (from cae.encoder)
    centroid: centroid vector (from certain samples in latent space)
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = RefinerAutoencoder(X_uncertain_latent.shape[1], latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_target = np.repeat(centroid[None, :], X_uncertain_latent.shape[0], axis=0)
    dataset = TensorDataset(to_tensor(X_uncertain_latent), to_tensor(X_target))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            z, x_rec = model(xb)
            loss = criterion(z, yb)  # Move latent to centroid
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Refiner Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    return model

def train_classifier(X_latent, y, epochs=20, batch_size=128, lr=1e-3):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Classifier(X_latent.shape[1], num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(to_tensor(X_latent), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Classifier Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    return model

# ---- Main Pipeline ----

# ---- Load and prepare initial training data ----
data_dir = '/home/mhaque3/myDir/data/gen_apigraph_drebin'
train_file = os.path.join(data_dir, '2012-01to2012-12_selected.npz')
train_npz = np.load(train_file)
X_train = train_npz['X_train']
y_train = binary_labels(train_npz['y_train'])

from sklearn.model_selection import train_test_split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# 1. Train contrastive autoencoder
latent_dim = 64
cae = train_contrastive_autoencoder(X_train_split, y_train_split, latent_dim=latent_dim)

# 2. Get latent representations for all training and validation data
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
cae.eval()
with torch.no_grad():
    z_train = cae.encoder(to_tensor(X_train).to(device)).detach().cpu().numpy()
    z_val = cae.encoder(to_tensor(X_val_split).to(device)).detach().cpu().numpy()

# 3. Find uncertain/certain samples in both train and val
centroids = get_centroids(z_train, y_train)
dists_train = np.linalg.norm(z_train - centroids[y_train], axis=1)
dists_val = np.linalg.norm(z_val - centroids[y_val_split], axis=1)

uncertainty_threshold_train = np.percentile(dists_train, 90)
uncertainty_threshold_val = np.percentile(dists_val, 70)

uncertain_idx_train = np.where(dists_train >= uncertainty_threshold_train)[0]
certain_idx_train = np.where(dists_train < uncertainty_threshold_train)[0]
uncertain_idx_val = np.where(dists_val >= uncertainty_threshold_val)[0]
certain_idx_val = np.where(dists_val < uncertainty_threshold_val)[0]

z_uncertain = np.concatenate([z_train[uncertain_idx_train], z_val[uncertain_idx_val]], axis=0)
y_uncertain = np.concatenate([y_train[uncertain_idx_train], y_val_split[uncertain_idx_val]], axis=0)
z_certain = np.concatenate([z_train[certain_idx_train], z_val[certain_idx_val]], axis=0)
y_certain = np.concatenate([y_train[certain_idx_train], y_val_split[certain_idx_val]], axis=0)

centroids_certain = get_centroids(z_certain, y_certain)

# ---- Aggregate all 2013 data for refiner training ----
X_2013 = []
y_2013 = []
for month in range(1, 13):
    fname = f"2013-{month:02d}_selected.npz"
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        npz = np.load(fpath)
        X_2013.append(npz['X_train'])
        y_2013.append(binary_labels(npz['y_train']))
if len(X_2013) > 0:
    X_2013 = np.concatenate(X_2013, axis=0)
    y_2013 = np.concatenate(y_2013, axis=0)
else:
    raise RuntimeError("No 2013 data found!")

# Get latent for 2013
cae.eval()
with torch.no_grad():
    z_2013 = cae.encoder(to_tensor(X_2013).to(device)).detach().cpu().numpy()

# Use centroids from initial certain pool
centroids_2013 = get_centroids(z_certain, y_certain)
dists_2013 = np.linalg.norm(z_2013 - centroids_2013[y_2013], axis=1)
uncertainty_threshold_2013 = np.percentile(dists_2013, 60)
uncertain_idx_2013 = np.where(dists_2013 >= uncertainty_threshold_2013)[0]
certain_idx_2013 = np.where(dists_2013 < uncertainty_threshold_2013)[0]

z_uncertain_2013 = z_2013[uncertain_idx_2013]
y_uncertain_2013 = y_2013[uncertain_idx_2013]
z_certain_2013 = z_2013[certain_idx_2013]
y_certain_2013 = y_2013[certain_idx_2013]

# Update certain pool with 2013 certain samples
z_certain = np.concatenate([z_certain, z_certain_2013], axis=0)
y_certain = np.concatenate([y_certain, y_certain_2013], axis=0)
centroids_certain = get_centroids(z_certain, y_certain)

# Update uncertain pool with 2013 uncertain samples
z_uncertain = np.concatenate([z_uncertain, z_uncertain_2013], axis=0)
y_uncertain = np.concatenate([y_uncertain, y_uncertain_2013], axis=0)

# ---- Train refiner(s) using 2013 uncertain samples ----
refiners = []
for label in [0, 1]:
    idx = np.where(y_uncertain == label)[0]
    if len(idx) > 0:
        refiner = train_refiner_autoencoder(z_uncertain[idx], centroids_certain[label], latent_dim=latent_dim)
        refiners.append(refiner)
    else:
        refiners.append(None)

# ---- Refine all uncertain samples (train/val/2013) ----
for label in [0, 1]:
    idx = np.where(y_uncertain == label)[0]
    if len(idx) > 0 and refiners[label] is not None:
        refiners[label].eval()
        with torch.no_grad():
            z_uncertain_refined, _ = refiners[label](to_tensor(z_uncertain[idx]).to(device))
            z_uncertain[idx] = z_uncertain_refined.detach().cpu().numpy()
    idx_2013 = np.where(y_uncertain_2013 == label)[0]
    if len(idx_2013) > 0 and refiners[label] is not None:
        refiners[label].eval()
        with torch.no_grad():
            z_uncertain_refined_2013, _ = refiners[label](to_tensor(z_uncertain_2013[idx_2013]).to(device))
            z_uncertain_2013[idx_2013] = z_uncertain_refined_2013.detach().cpu().numpy()

# ---- Prepare final training pool for classifier ----
z_final = np.concatenate([z_certain, z_uncertain, z_uncertain_2013], axis=0)
y_final = np.concatenate([y_certain, y_uncertain, y_uncertain_2013], axis=0)

# ---- Train classifier on final latent space ----
clf = train_classifier(z_final, y_final)

print("Classifier metrics on training data:")
logits = clf(to_tensor(z_final).to(device)).detach().cpu()
y_pred = logits.argmax(axis=1).numpy()
y_score = torch.softmax(logits, dim=1)[:, 1].numpy()
metrics = evaluate_metrics(y_final, y_pred, y_score)
print(metrics)


print("Classifier metrics on all combined data (train + val + 2013):")
z_combined = np.concatenate([z_train, z_val, z_2013], axis=0)
y_combined = np.concatenate([y_train, y_val_split, y_2013], axis=0)
logits = clf(to_tensor(z_combined).to(device)).detach().cpu()
y_pred = logits.argmax(axis=1).numpy()
y_score = torch.softmax(logits, dim=1)[:, 1].numpy()
metrics_combined = evaluate_metrics(y_combined, y_pred, y_score)
print(metrics_combined)

# ---- Monthwise testing from 2014 onward (NO retraining) ----
results = []
for year in range(2014, 2019):
    for month in range(1, 13):
        fname = f"{year}-{month:02d}_selected.npz"
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        npz = np.load(fpath)
        X_test = npz['X_train']
        y_test = binary_labels(npz['y_train'])

        # Get latent for test
        cae.eval()
        with torch.no_grad():
            z_test = cae.encoder(to_tensor(X_test).to(device)).detach().cpu().numpy()

        # Use fixed centroids from initial certain pool
        centroids_certain = get_centroids(z_certain, y_certain)

        # Use classifier to get initial predictions
        clf.eval()
        with torch.no_grad():
            logits = clf(to_tensor(z_test).to(device)).detach().cpu().numpy()
            y_pred = logits.argmax(axis=1)
            y_score = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

        # Separate certain/uncertain in test set for drift adaptation (using current centroids)
        dists_test = np.linalg.norm(z_test - centroids_certain[y_pred], axis=1)
        test_uncertainty_threshold = np.percentile(dists_test, 90)
        test_certain_idx = np.where(dists_test < test_uncertainty_threshold)[0]
        test_uncertain_idx = np.where(dists_test >= test_uncertainty_threshold)[0]

        # --- Apply refiner(s) to uncertain latent samples before final prediction ---
        if len(test_uncertain_idx) > 0:
            z_uncertain_new = z_test[test_uncertain_idx]
            y_uncertain_pred = y_pred[test_uncertain_idx]
            # Refine uncertain latent samples
            for label in [0, 1]:
                idx = np.where(y_uncertain_pred == label)[0]
                if len(idx) > 0 and refiners[label] is not None:
                    refiners[label].eval()
                    with torch.no_grad():
                        z_uncertain_refined, _ = refiners[label](to_tensor(z_uncertain_new[idx]).to(device))
                        z_uncertain_new[idx] = z_uncertain_refined.detach().cpu().numpy()
            # Update z_test with refined uncertain samples
            z_test[test_uncertain_idx] = z_uncertain_new

        # --- Now do classifier prediction on refined z_test ---
        clf.eval()
        with torch.no_grad():
            logits = clf(to_tensor(z_test).to(device)).detach().cpu().numpy()
            y_pred = logits.argmax(axis=1)
            y_score = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

        # Evaluate and save
        metrics = evaluate_metrics(y_test, y_pred, y_score)
        metrics.update({"year": year, "month": month, "num_samples": len(y_test)})
        results.append(metrics)
        print(f"{year}-{month:02d}: {metrics}")

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("concept_drift_monthwise_results.csv", index=False)
print("Saved results to concept_drift_monthwise_results.csv")

















# data_dir = '/home/mhaque3/myDir/data/gen_apigraph_drebin'
# train_file = os.path.join(data_dir, '2012-01to2012-12_selected.npz')
# train_npz = np.load(train_file)
# X_train = train_npz['X_train']
# y_train = binary_labels(train_npz['y_train'])

# # Split for initial contrastive autoencoder training
# from sklearn.model_selection import train_test_split
# X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
#     X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
# )



# # 1. Train contrastive autoencoder
# latent_dim = 64
# cae = train_contrastive_autoencoder(X_train_split, y_train_split, latent_dim=latent_dim)

# # 2. Get latent representations for all training and validation data
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# cae.eval()
# with torch.no_grad():
#     z_train = cae.encoder(to_tensor(X_train).to(device)).cpu().numpy()
#     z_val = cae.encoder(to_tensor(X_val_split).to(device)).cpu().numpy()

# # 3. Find uncertain/certain samples in both train and val
# centroids = get_centroids(z_train, y_train)
# dists_train = np.linalg.norm(z_train - centroids[y_train], axis=1)
# dists_val = np.linalg.norm(z_val - centroids[y_val_split], axis=1)

# # Use 80th percentile for uncertainty threshold
# uncertainty_threshold_train = np.percentile(dists_train, 80)
# uncertainty_threshold_val = np.percentile(dists_val, 80)

# uncertain_idx_train = np.where(dists_train >= uncertainty_threshold_train)[0]
# certain_idx_train = np.where(dists_train < uncertainty_threshold_train)[0]
# uncertain_idx_val = np.where(dists_val >= uncertainty_threshold_val)[0]
# certain_idx_val = np.where(dists_val < uncertainty_threshold_val)[0]

# # Combine uncertain samples from train and val
# z_uncertain = np.concatenate([z_train[uncertain_idx_train], z_val[uncertain_idx_val]], axis=0)
# y_uncertain = np.concatenate([y_train[uncertain_idx_train], y_val_split[uncertain_idx_val]], axis=0)

# # Combine certain samples from train and val
# z_certain = np.concatenate([z_train[certain_idx_train], z_val[certain_idx_val]], axis=0)
# y_certain = np.concatenate([y_train[certain_idx_train], y_val_split[certain_idx_val]], axis=0)

# # Calculate centroid from certain samples only
# centroids_certain = get_centroids(z_certain, y_certain)

# # 4. Train refiner autoencoder (move uncertain to centroid of their class)
# # Here, you can train one refiner per class or a single refiner for all
# refiners = []
# for label in [0, 1]:
#     idx = np.where(y_uncertain == label)[0]
#     if len(idx) > 0:
#         refiner = train_refiner_autoencoder(z_uncertain[idx], centroids_certain[label], latent_dim=latent_dim)
#         refiners.append(refiner)
#     else:
#         refiners.append(None)

# # 5. Get refined latent space for all data
# cae.eval()
# for label in [0, 1]:
#     idx = np.where(y_uncertain == label)[0]
#     if len(idx) > 0 and refiners[label] is not None:
#         refiners[label].eval()
#         with torch.no_grad():
#             z_uncertain_refined, _ = refiners[label](to_tensor(z_uncertain[idx]).to(device))
#             z_uncertain[idx] = z_uncertain_refined.cpu().numpy()

# # Now, z_certain and z_uncertain (refined) can be concatenated for classifier training
# z_final = np.concatenate([z_certain, z_uncertain], axis=0)
# y_final = np.concatenate([y_certain, y_uncertain], axis=0)

# # 6. Train classifier on final latent space
# clf = train_classifier(z_final, y_final)

# # 7. Monthwise testing and drift adaptation
# results = []
# X_certain_all = X_train[certain_idx_train]
# y_certain_all = y_train[certain_idx_train]

# # Iterate over each month from 2013 to 2015
# for year in range(2013, 2016):
#     for month in range(1, 13):
#         fname = f"{year}-{month:02d}_selected.npz"
#         fpath = os.path.join(data_dir, fname)
#         if not os.path.exists(fpath):
#             continue
#         npz = np.load(fpath)
#         X_test = npz['X_train']
#         y_test = binary_labels(npz['y_train'])

#         # Get latent for test
#         cae.eval()
#         with torch.no_grad():
#             z_test = cae.encoder(to_tensor(X_test).to(device)).cpu().numpy()

#         # Recompute centroids from current certain pool in latent space
#         with torch.no_grad():
#             z_certain_all = cae.encoder(to_tensor(X_certain_all).to(device)).detach().cpu().numpy()
#         centroids_certain = get_centroids(z_certain_all, y_certain_all)

#         # Use previous classifier to get initial predictions
#         clf.eval()
#         with torch.no_grad():
#             logits = clf(to_tensor(z_test).to(device)).detach().cpu().numpy()
#             y_pred = logits.argmax(axis=1)
#             y_score = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

#         # Separate certain/uncertain in test set for drift adaptation (using current centroids)
#         dists_test = np.linalg.norm(z_test - centroids_certain[y_pred], axis=1)
#         test_uncertainty_threshold = np.percentile(dists_test, 90)
#         test_certain_idx = np.where(dists_test < test_uncertainty_threshold)[0]
#         test_uncertain_idx = np.where(dists_test >= test_uncertainty_threshold)[0]

#         # --- Apply refiner(s) to uncertain latent samples before final prediction ---
#         if len(test_uncertain_idx) > 0:
#             z_uncertain_new = z_test[test_uncertain_idx]
#             y_uncertain_pred = y_pred[test_uncertain_idx]
#             logits_uncertain = logits[test_uncertain_idx]

#             # Compute uncertainty scores (entropy)
#             scores_uncertain = uncertainty_scores(logits_uncertain, method='entropy')
#             # Exclude extremely uncertain samples (top 10% by entropy)
#             extreme_threshold = np.percentile(scores_uncertain, 90)
#             not_extreme_idx = np.where(scores_uncertain < extreme_threshold)[0]

#             # Filter out extremely uncertain samples
#             z_uncertain_filtered = z_uncertain_new[not_extreme_idx]
#             y_uncertain_filtered = y_uncertain_pred[not_extreme_idx]

#             # Refine uncertain latent samples (filtered only)
#             for label in [0, 1]:
#                 idx = np.where(y_uncertain_filtered == label)[0]
#                 if len(idx) > 0 and refiners[label] is not None:
#                     refiners[label].eval()
#                     with torch.no_grad():
#                         z_uncertain_refined, _ = refiners[label](to_tensor(z_uncertain_filtered[idx]).to(device))
#                         z_uncertain_filtered[idx] = z_uncertain_refined.detach().cpu().numpy()
#             # Update z_test with refined uncertain samples (filtered only)
#             z_test[test_uncertain_idx[not_extreme_idx]] = z_uncertain_filtered

#         # --- Now do classifier prediction on refined z_test ---
#         clf.eval()
#         with torch.no_grad():
#             logits = clf(to_tensor(z_test).to(device)).detach().cpu().numpy()
#             y_pred = logits.argmax(axis=1)
#             y_score = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

#         # Evaluate and save
#         metrics = evaluate_metrics(y_test, y_pred, y_score)
#         metrics.update({"year": year, "month": month, "num_samples": len(y_test)})
#         results.append(metrics)
#         print(f"{year}-{month:02d}: {metrics}")

#         # Add certain samples to training pool (use only X and predicted labels)
#         X_certain_all = np.concatenate([X_certain_all, X_test[test_certain_idx]], axis=0)
#         y_certain_all = np.concatenate([y_certain_all, y_pred[test_certain_idx]], axis=0)

#         # Use uncertain samples for refiner retraining (no y_test used)
#         if len(test_uncertain_idx) > 0:
#             z_uncertain_new = z_test[test_uncertain_idx]
#             y_uncertain_pred = y_pred[test_uncertain_idx]
#             logits_uncertain = logits[test_uncertain_idx]

#             # Compute uncertainty scores (entropy)
#             scores_uncertain = uncertainty_scores(logits_uncertain, method='entropy')
#             # Exclude extremely uncertain samples (top 10% by entropy)
#             extreme_threshold = np.percentile(scores_uncertain, 90)
#             not_extreme_idx = np.where(scores_uncertain < extreme_threshold)[0]

#             # Filter out extremely uncertain samples
#             z_uncertain_filtered = z_uncertain_new[not_extreme_idx]
#             y_uncertain_filtered = y_uncertain_pred[not_extreme_idx]

#             # Retrain one refiner per class (using only filtered uncertain samples)
#             refiners = []
#             for label in [0, 1]:
#                 idx = np.where(y_uncertain_filtered == label)[0]
#                 if len(idx) > 0:
#                     refiner = train_refiner_autoencoder(z_uncertain_filtered[idx], centroids_certain[label], latent_dim=latent_dim)
#                     refiners.append(refiner)
#                 else:
#                     refiners.append(None)
#             # Refine uncertain latent samples for next round (filtered only)
#             for label in [0, 1]:
#                 idx = np.where(y_uncertain_filtered == label)[0]
#                 if len(idx) > 0 and refiners[label] is not None:
#                     refiners[label].eval()
#                     with torch.no_grad():
#                         z_uncertain_refined, _ = refiners[label](to_tensor(z_uncertain_filtered[idx]).to(device))
#                         z_uncertain_filtered[idx] = z_uncertain_refined.detach().cpu().numpy()
#             # Update latent space for classifier retraining
#             z_certain_all = cae.encoder(to_tensor(X_certain_all).to(device)).detach().cpu().numpy()
#             z_final_all = np.concatenate([z_certain_all, z_uncertain_filtered], axis=0)
#             y_final_all = np.concatenate([y_certain_all, y_uncertain_filtered], axis=0)
#             clf = train_classifier(z_final_all, y_final_all)

# # Save results
# df_results = pd.DataFrame(results)
# df_results.to_csv("concept_drift_monthwise_results.csv", index=False)
# print("Saved results to concept_drift_monthwise_results.csv")
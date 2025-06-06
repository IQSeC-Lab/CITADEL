import torch
from sklearn.metrics import f1_score, confusion_matrix
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
import plot_gen

def model_evaluate(model, test_sets_by_year, strategy):
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for year, (X_test, y_test) in test_sets_by_year.items():
            X_test, y_test = X_test.cuda(), y_test.cuda()
            logits = model(X_test)
            probs = torch.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)
            preds = logits.argmax(dim=1)
            y_true = y_test.cpu().numpy()
            y_pred = preds.cpu().numpy()
            if probs.shape[1] == 2:
                y_score = probs[:, 1].cpu().numpy()
            else:
                y_score = probs.cpu().numpy()  # for multi-class

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            else:
                fnr = fpr = float('nan')

            # ROC-AUC and PR-AUC (binary or multiclass)
            try:
                if probs.shape[1] == 2:
                    roc_auc = roc_auc_score(y_true, y_score)
                    pr_auc = average_precision_score(y_true, y_score)
                else:
                    roc_auc = roc_auc_score(y_true, probs.cpu().numpy(), multi_class='ovr')
                    pr_auc = average_precision_score(y_true, probs.cpu().numpy(), average='weighted')
            except Exception:
                roc_auc = pr_auc = float('nan')

            metrics_list.append({
                'year': year,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'fnr': fnr,
                'fpr': fpr,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            })

            print(f"Year {year}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, FNR={fnr:.4f}, FPR={fpr:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

    # Save results to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(f"/home/ihossain/ISMAIL/SSL-malware/baseline_experiments/flexmatch/results/{strategy}.csv", index=False)

    print(f"Mean F1 Scores: {metrics_df['f1'].mean():.4f}")
    print(f"Mean False Negative Rates: {metrics_df['fnr'].mean()}")
    print(f"Mean False Positive Rates: {metrics_df['fpr'].mean()}")
    plot_gen.plot_f1_fnr(metrics_df['year'], metrics_df['f1'], metrics_df['fnr'], save_path=f"results/{strategy}_f1_fnr_plot.png")



def evaluate_model_active(model, X_test, y_test, num_classes=2):
    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.cuda(), y_test.cuda()
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)
        preds = logits.argmax(dim=1)
        y_true = y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()
        if probs.shape[1] == 2:
            y_score = probs[:, 1].cpu().numpy()
        else:
            y_score = probs.cpu().numpy()  # for multi-class

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fnr = fpr = float('nan')

        # ROC-AUC and PR-AUC (binary or multiclass)
        try:
            if probs.shape[1] == 2:
                roc_auc = roc_auc_score(y_true, y_score)
                pr_auc = average_precision_score(y_true, y_score)
            else:
                roc_auc = roc_auc_score(y_true, probs.cpu().numpy(), multi_class='ovr')
                pr_auc = average_precision_score(y_true, probs.cpu().numpy(), average='weighted')
        except Exception:
            roc_auc = pr_auc = float('nan')

        metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'fnr': fnr,
            'fpr': fpr,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        return metrics
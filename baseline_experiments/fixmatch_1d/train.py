import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset.randaugment import RandAugmentMC1D

from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0


# === Classifier Definition ===
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 384), nn.ReLU(),
            nn.Linear(384, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, num_classes)
            #nn.Linear(100, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(self.encoder(x))

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def split_labeled_unlabeled(X, y, labeled_ratio=0.1, stratify=True, random_state=42):
    n_samples = len(X)
    n_labeled = int(n_samples * labeled_ratio)
    if stratify:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, train_size=n_labeled, stratify=y, random_state=random_state
        )
    else:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, train_size=n_labeled, random_state=random_state
        )
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled


class TransformFixMatch1D:
    def __init__(self, n=2, m=5):
        self.strong_aug = RandAugmentMC1D(n=n, m=m)

    def weak_augment(self, x):
        # Weak = Identity (no-op) or minor noise
        return x.clone()

    def strong_augment(self, x):
        return self.strong_aug(x.clone())

    def __call__(self, x):
        weak = self.weak_augment(x)
        strong = self.strong_augment(x)
        return weak, strong



def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=52, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    global best_acc

    

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    path = "/home/mhaque3/myDir/data/gen_apigraph_drebin/"
    file_path = f"{path}2012-01to2012-12_selected.npz"
    data = np.load(file_path, allow_pickle=True)
    X, y = data['X_train'], data['y_train']
    y = np.array([0 if label == 0 else 1 for label in y])

    # Use the labeled_ratio argument here!
    X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, labeled_ratio=0.4, random_state=args.seed)

    X_2012_labeled = torch.tensor(X_labeled, dtype=torch.float32).cuda()
    y_2012_labeled = torch.tensor(y_labeled, dtype=torch.long).cuda()
    X_2012_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).cuda()
    from torch.utils.data import TensorDataset
    labeled_dataset = TensorDataset(X_2012_labeled, y_2012_labeled)
    unlabeled_dataset = TensorDataset(X_2012_unlabeled)

    input_dim = X_2012_labeled.shape[1]
    num_classes = len(torch.unique(y_2012_labeled))    

    
    

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=0,
        drop_last=True)

    X_test, y_test = [], []
    for year in range(2013, 2019):
        for month in range(1, 13):
            try:
                data = np.load(f"{path}{year}-{month:02d}_selected.npz")
                X_raw = data["X_train"]
                y_true = (data["y_train"] > 0).astype(int)
                X_tensor = torch.tensor(X_raw, dtype=torch.float32).cuda()
                y_tensor = torch.tensor(y_true, dtype=torch.long).cuda()
                X_test.append(X_tensor)
                y_test.append(y_tensor)
            except FileNotFoundError:
                continue
    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)
    # del X_test, y_test
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=0)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = Classifier(input_dim=input_dim, num_classes=num_classes).cuda()

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    # args.epochs = 1
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            transform = TransformFixMatch1D(n=2, m=5)

            try:
                (inputs_u_raw,) = next(unlabeled_iter)
            except StopIteration:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_raw,) = next(unlabeled_iter)

            inputs_u_raw = inputs_u_raw.to(args.device)
            inputs_u_w, inputs_u_s = transform(inputs_u_raw)


            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    path = "/home/mhaque3/myDir/data/gen_apigraph_drebin/"
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for year in range(2013, 2019):
            for month in range(1, 13):
                try:
                    data = np.load(f"{path}{year}-{month:02d}_selected.npz")
                    X_raw = data["X_train"]
                    y_true = (data["y_train"] > 0).astype(int)
                    X_test = torch.tensor(X_raw, dtype=torch.float32).cuda()
                    y_test = torch.tensor(y_true, dtype=torch.long).cuda()

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
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
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
                        'year': f"{year}_{month}",
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'fnr': fnr,
                        'fpr': fpr,
                        'roc_auc': roc_auc,
                        'pr_auc': pr_auc
                    })

                    print(f"Year {year}_{month}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, FNR={fnr:.4f}, FPR={fpr:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

                except FileNotFoundError:
                    continue

    # Save results to CSV
    strategy = f"fixmatch_baseline_keeping_as_it_is_epoch_{args.epochs}_seed_{args.seed}"
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(f"/home/mhaque3/myDir/SSL-malware/baseline_experiments/fixmatch_1d/{strategy}.csv", index=False)

    print(f"Mean F1 Scores: {metrics_df['f1'].mean():.4f}")
    print(f"Mean False Negative Rates: {metrics_df['fnr'].mean()}")
    print(f"Mean False Positive Rates: {metrics_df['fpr'].mean()}")
    plot_f1_fnr(metrics_df['year'], metrics_df['f1'], metrics_df['fnr'], save_path=f"/home/mhaque3/myDir/SSL-malware/baseline_experiments/fixmatch_1d/{strategy}_f1_fnr_plot.png")
    

    if args.local_rank in [-1, 0]:
        args.writer.close()


def plot_f1_fnr(years, f1s, fnrs, save_path="f1_fnr_fixmatch_baseline_with_al.png"):
    # Convert to list if Series
    years = list(years)
    f1s = list(f1s)
    fnrs = list(fnrs)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Year")
    ax1.set_ylabel("F1 Score", color="blue")
    ax1.plot(years, f1s, color="blue", label="F1 Score")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(0, 1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel("False Negative Rate (FNR)", color="red")
    ax2.plot(years, fnrs, color="red", label="FNR")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1)

    # Set only year (4-digit) on x-axis, sampled to reduce overlap
    xtick_positions = []
    xtick_labels = []
    seen_years = set()
    for idx, ym in enumerate(years):
        year = ym.split("_")[0]
        if year not in seen_years:
            xtick_positions.append(idx)
            xtick_labels.append(year)
            seen_years.add(year)

    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(xtick_labels, rotation=0)

    # Add legend above plot to avoid overlapping x-label
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.show()




def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            # top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    # logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()

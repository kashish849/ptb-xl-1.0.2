"""
PTB-XL: MI vs NORM binary classification  --  3 models

  cnn     : plain 4-layer CNN (baseline)
  resnet  : ResNet1D
  hybrid  : ResNet + Inception + Transformer encoder  (proposed model)

Extras: official split (folds 1-8 / 9 / 10), AMP, OneCycleLR, pos_weight,
        threshold tuning (on val), seed ensemble, TTA, comparison table,
        ROC curves, Grad-CAM.

Install : pip install torch wfdb pandas numpy scikit-learn matplotlib tqdm  (torch >= 2.3)
Run     : python ptbxl_3models.py --data_dir /path/to/ptb-xl-1.0.2/
"""
import argparse
import ast
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODELS = ["cnn", "resnet", "hybrid"]

# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------


def load_data(data_dir, sampling_rate=100, min_likelihood=100, cache_dir="cache", label_mode="pure"):
    os.makedirs(cache_dir, exist_ok=True)
    Y = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"), index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

    agg = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
    agg = agg[agg.diagnostic == 1]

    def to_super(codes):
        s = set()
        for k, lik in codes.items():
            if k in agg.index and lik >= min_likelihood:
                s.add(agg.loc[k].diagnostic_class)
        return s

    Y["super"] = Y.scp_codes.apply(to_super)
    # label_mode:
    #   pure       : only pure NORM (0) vs pure MI (1)
    #   mi_any     : NORM (pure) = 0, any record containing MI (even with other conditions) = 1
    #   mi_vs_rest : ALL records. MI (any) = 1, everything else (NORM/STTC/CD/HYP/other) = 0
    if label_mode == "pure":
        Y = Y[Y["super"].apply(lambda s: s == {"NORM"} or s == {"MI"})].copy()
    elif label_mode == "mi_any":
        Y = Y[Y["super"].apply(lambda s: s == {"NORM"} or "MI" in s)].copy()
    else:  # mi_vs_rest: keep ALL records (every record without MI is class 0)
        Y = Y.copy()
    Y["label"] = Y["super"].apply(lambda s: 1 if "MI" in s else 0)
    print(f"Records kept: {len(Y)} | NORM: {(Y.label == 0).sum()} | MI: {(Y.label == 1).sum()}")

    cache_file = os.path.join(cache_dir, f"X_{sampling_rate}_{min_likelihood}_{label_mode}_{len(Y)}.npy")
    if os.path.exists(cache_file):
        X = np.load(cache_file)
    else:
        col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        X = np.stack([wfdb.rdsamp(os.path.join(data_dir, f))[0]
                      for f in tqdm(Y[col], desc="Loading ECGs")]).astype(np.float32)
        np.save(cache_file, X)

    X = np.nan_to_num(X)
    y = Y.label.values.astype(np.float32)
    folds = Y.strat_fold.values
    X = X.transpose(0, 2, 1)  # (N, 12, T) channels-first for Conv1d

    tr, va, te = folds <= 8, folds == 9, folds == 10
    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std = X[tr].std(axis=(0, 2), keepdims=True) + 1e-8
    X = (X - mean) / std  # statistics from train only (no leakage)
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


class ECGDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            x = x * np.random.uniform(0.9, 1.1)                          # amplitude scaling
            x = x + np.random.normal(0, 0.05, x.shape).astype(np.float32)  # noise
            x = np.roll(x, np.random.randint(-50, 51), axis=1)           # time shift
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(self.y[i])


# ----------------------------------------------------------------------------
# 2. MODEL BLOCKS
# ----------------------------------------------------------------------------


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(c // r, 4))
        self.fc2 = nn.Linear(max(c // r, 4), c)

    def forward(self, x):
        s = x.mean(dim=2)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s)))).unsqueeze(2)
        return x * s


class ResBlock(nn.Module):
    def __init__(self, cin, cout, k=15, stride=1, drop=0.2, se=True):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = SE(cout) if se else nn.Identity()
        self.drop = nn.Dropout(drop)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm1d(cout))

    def forward(self, x):
        out = self.drop(F.relu(self.b1(self.c1(x))))
        out = self.se(self.b2(self.c2(out)))
        return F.relu(out + (x if self.short is None else self.short(x)))


class InceptionBlock(nn.Module):
    """Multi-scale block: parallel convs with different kernel sizes + pooling branch."""

    def __init__(self, cin, nf=64, ks=(9, 19, 39), stride=1, drop=0.2):
        super().__init__()
        self.bottleneck = nn.Conv1d(cin, nf, 1, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(nf, nf, k, stride, k // 2, bias=False) for k in ks])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride, 1), nn.Conv1d(cin, nf, 1, bias=False))
        cout = nf * (len(ks) + 1)
        self.bn = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(drop)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm1d(cout))

    def forward(self, x):
        b = self.bottleneck(x)
        out = torch.cat([c(b) for c in self.convs] + [self.pool(x)], dim=1)
        out = self.drop(F.relu(self.bn(out)))
        return F.relu(out + (x if self.short is None else self.short(x)))


def sinusoidal_pe(T, d, device):
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    i = torch.arange(0, d, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerHead(nn.Module):
    def __init__(self, d=256, heads=4, layers=2, ff=512, drop=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, f):  # f: (B, d, T')
        x = f.transpose(1, 2)
        x = x + sinusoidal_pe(x.shape[1], self.d, x.device).to(x.dtype)
        return self.norm(self.enc(x)).mean(dim=1)


# ----------------------------------------------------------------------------
# 3. THE 3 MODELS
# ----------------------------------------------------------------------------

RES_CFG = [(32, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1)]


class ECGNet(nn.Module):
    def __init__(self, variant="hybrid", in_ch=12, k=15, drop=0.2):
        super().__init__()
        assert variant in MODELS, f"unknown model {variant}"

        def stem():
            return nn.Sequential(nn.Conv1d(in_ch, 32, k, 2, k // 2, bias=False),
                                 nn.BatchNorm1d(32), nn.ReLU())

        def res(cfg, se):
            return [ResBlock(a, b, k, s, drop, se) for a, b, s in cfg]

        if variant == "cnn":
            def cb(a, b):
                return [nn.Conv1d(a, b, k, 2, k // 2, bias=False), nn.BatchNorm1d(b), nn.ReLU(),
                        nn.Dropout(drop)]
            layers = cb(in_ch, 32) + cb(32, 64) + cb(64, 128) + cb(128, 256)
        elif variant == "resnet":
            layers = [stem()] + res(RES_CFG, se=False)
        else:  # hybrid: ResNet(+SE) -> Inception -> Transformer
            layers = [stem()] + res(RES_CFG[:4], se=True) + [
                InceptionBlock(128, 64, stride=1, drop=drop),
                InceptionBlock(256, 64, stride=2, drop=drop)]

        self.features = nn.Sequential(*layers)  # -> (B, 256, T')
        self.tf = TransformerHead(256) if variant == "hybrid" else None
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 1))

    def head(self, f):  # f: (B, 256, T')
        z = self.tf(f) if self.tf is not None else f.mean(dim=2)
        return self.fc(z).squeeze(1)

    def forward(self, x):
        return self.head(self.features(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------
# 4. TRAIN / EVAL
# ----------------------------------------------------------------------------


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x))
        if tta:  # tiny test-time augmentation: two small shifts
            p = (p + torch.sigmoid(model(torch.roll(x, 20, 2))) +
                 torch.sigmoid(model(torch.roll(x, -20, 2)))) / 3
        probs.append(p.float().cpu())
    return torch.cat(probs).numpy()


def train_one(name, seed, data, args, device):
    set_seed(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    tr_dl = DataLoader(ECGDataset(Xtr, ytr, True), batch_size=args.bs, shuffle=True,
                       num_workers=2, drop_last=True)
    va_dl = DataLoader(ECGDataset(Xva, yva), batch_size=256)
    te_dl = DataLoader(ECGDataset(Xte, yte), batch_size=256)

    model = ECGNet(name).to(device)
    print(f"\n[{name}] seed {seed} | params: {count_params(model) / 1e6:.2f}M")

    pos_weight = torch.tensor(float((ytr == 0).sum() / (ytr == 1).sum()), device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                steps_per_epoch=len(tr_dl), epochs=args.epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc, best_state, bad = 0, None, 0
    for ep in range(args.epochs):
        model.train()
        total = 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        val_p = predict(model, va_dl, device)
        auc = roc_auc_score(yva, val_p)
        acc = accuracy_score(yva, val_p > 0.5)
        print(f"Ep {ep + 1:3d} | loss {total / len(tr_dl):.4f} | val AUC {auc:.4f} | val acc {acc:.4f}")
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, f"best_{name}_seed{seed}.pt")
    return model, predict(model, va_dl, device, tta=True), predict(model, te_dl, device, tta=True)


def best_threshold(y, p):
    ths = np.linspace(0.05, 0.95, 91)
    return ths[int(np.argmax([f1_score(y, p > t) for t in ths]))]


def best_threshold_acc(y, p):
    ths = np.linspace(0.05, 0.95, 91)
    return ths[int(np.argmax([accuracy_score(y, p > t) for t in ths]))]


def metrics(y, p, th):
    pred = (p > th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, pred), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                f1=f1_score(y, pred), auc=roc_auc_score(y, p), cm=np.array([[tn, fp], [fn, tp]]))


def report(m, th, title):
    print(f"\n===== {title} (threshold {th:.2f}) =====")
    print(f"Accuracy {m['acc']:.4f} | Sensitivity {m['sens']:.4f} | Specificity {m['spec']:.4f} "
          f"| F1 {m['f1']:.4f} | AUC {m['auc']:.4f}")
    print(f"Confusion matrix [[TN FP],[FN TP]]:\n{m['cm']}")


# ----------------------------------------------------------------------------
# 5. GRAD-CAM
# ----------------------------------------------------------------------------


def grad_cam(model, x, device):
    """x: (12, T) numpy. Returns CAM (T,) normalised to 0-1."""
    model.eval()
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    feats = model.features(xt)
    feats.retain_grad()
    out = model.head(feats)
    out.backward()
    w = feats.grad.mean(dim=2, keepdim=True)
    cam = F.relu((w * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[1], mode="linear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def plot_gradcam(model, X, y, device, lead=1, fname="gradcam.png"):
    idx = np.where(y == 1)[0][:3]  # first 3 MI test samples
    fig, axes = plt.subplots(len(idx), 1, figsize=(10, 3 * len(idx)))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idx):
        cam = grad_cam(model, X[i], device)
        t = np.arange(X.shape[2])
        ax.plot(t, X[i, lead], "k", lw=0.8)
        ax.imshow(cam[None, :], aspect="auto", cmap="jet", alpha=0.35,
                  extent=[0, len(t), X[i, lead].min(), X[i, lead].max()])
        ax.set_title(f"MI sample {i} | lead index {lead}")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--sr", type=int, default=100, choices=[100, 500])
    ap.add_argument("--min_likelihood", type=int, default=50)
    ap.add_argument("--label_mode", default="pure", choices=["pure", "mi_any", "mi_vs_rest"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    data = load_data(args.data_dir, args.sr, args.min_likelihood, label_mode=args.label_mode)
    yva, yte = data[1][1], data[2][1]

    rows, rocs, first_models = [], {}, {}
    for name in args.models:
        vps, tps, ms = [], [], []
        for s in args.seeds:
            m, vp, tp = train_one(name, s, data, args, device)
            report(metrics(yte, tp, 0.5), 0.5, f"{name} seed {s} (test)")
            ms.append(m)
            vps.append(vp)
            tps.append(tp)

        val_ens, test_ens = np.mean(vps, 0), np.mean(tps, 0)
        th = best_threshold(yva, val_ens)  # threshold chosen on VAL, applied on TEST
        th_acc = best_threshold_acc(yva, val_ens)  # accuracy-optimal threshold, also from VAL
        m05, mth, mac = (metrics(yte, test_ens, 0.5), metrics(yte, test_ens, th),
                         metrics(yte, test_ens, th_acc))
        report(m05, 0.5, f"{name} (test)")
        report(mth, th, f"{name} + F1-tuned threshold (test)")
        report(mac, th_acc, f"{name} + accuracy-tuned threshold (test)")

        rows.append({"model": name, "params_M": round(count_params(ms[0]) / 1e6, 2),
                     "acc@0.5": round(m05["acc"], 4), "acc@F1thr": round(mth["acc"], 4),
                     "acc@ACCthr": round(mac["acc"], 4),
                     "sens@ACCthr": round(mac["sens"], 4), "spec@ACCthr": round(mac["spec"], 4),
                     "f1@ACCthr": round(mac["f1"], 4), "auc": round(mth["auc"], 4)})
        pd.DataFrame(rows).to_csv("results.csv", index=False)  # saved after every model
        fpr, tpr, _ = roc_curve(yte, test_ens)
        rocs[name] = (fpr, tpr, mth["auc"])
        first_models[name] = ms[0]

    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)
    print("\n================ RESULTS (test set) ================")
    print(df.to_string(index=False))

    plt.figure(figsize=(6, 6))
    for name, (fpr, tpr, auc) in rocs.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: MI vs NORM")
    plt.legend(loc="lower right")
    plt.savefig("roc.png", dpi=150)
    plt.close()

    gname = "hybrid" if "hybrid" in first_models else list(first_models)[-1]
    plot_gradcam(first_models[gname], data[2][0], yte, device)
    print(f"\nSaved: results.csv, roc.png, gradcam.png ({gname}), best_<model>_seed*.pt")
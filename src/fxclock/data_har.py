"""UCI HAR (Reyes-Ortiz et al.), raw inertial signals: 9 channels, 50 Hz, 128-sample windows.

Source: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
(sha256 of the outer zip recorded in configs/prereg_n2.json).  License: CC BY 4.0 (UCI).
The official split is subject-disjoint (21 train / 9 test subjects).  A dev split is carved
out of the TRAIN subjects only (subject-disjoint), chosen by a fixed seed before any run.
"""

import os

import numpy as np

CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z"]
NATIVE_DT = 1.0 / 50.0


def load_split(root, split):
    sig = [np.loadtxt(os.path.join(root, split, "Inertial Signals", f"{c}_{split}.txt")) for c in CHANNELS]
    X = np.stack(sig, axis=-1)                                  # (n, 128, 9)
    y = np.loadtxt(os.path.join(root, split, f"y_{split}.txt")).astype(np.int64) - 1
    s = np.loadtxt(os.path.join(root, split, f"subject_{split}.txt")).astype(np.int64)
    return X, y, s


def load(root, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    Xtr, ytr, str_ = load_split(root, "train")
    Xte, yte, ste = load_split(root, "test")
    d = {"Xtr": Xtr, "ytr": ytr, "str": str_, "Xte": Xte, "yte": yte, "ste": ste}
    np.savez(cache, **d)
    return d


def dev_subjects(train_subjects, n_dev, seed):
    subs = np.unique(train_subjects)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(subs, size=n_dev, replace=False))

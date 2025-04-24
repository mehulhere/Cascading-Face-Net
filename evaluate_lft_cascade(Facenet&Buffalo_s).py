#!/usr/bin/env python3
"""
evaluate_lfw_cascade.py

Evaluate LFW accuracy for Facenet, EdgeFace, and a cascading architecture
using lab-tested pipelines to preserve SOTA numbers without stray resizing or
whitening missteps.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import insightface
import cv2 # For BGR image loading
from torchvision import datasets
from torch.utils.data import DataLoader, SequentialSampler

def load_pairs(pairs_path, lfw_dir):
    pairs, labels = [], []
    with open(pairs_path, 'r') as f:
        # Skip header line (<num_folds> <pairs_per_fold>)
        f.readline()
        # Read all pairs lines
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                # same person
                name, idx1, idx2 = parts
                p1 = os.path.join(lfw_dir, name, f"{name}_{int(idx1):04d}.jpg")
                p2 = os.path.join(lfw_dir, name, f"{name}_{int(idx2):04d}.jpg")
                pairs.append((p1, p2)); labels.append(True)
            elif len(parts) == 4:
                # different persons
                name1, idx1, name2, idx2 = parts
                p1 = os.path.join(lfw_dir, name1, f"{name1}_{int(idx1):04d}.jpg")
                p2 = os.path.join(lfw_dir, name2, f"{name2}_{int(idx2):04d}.jpg")
                pairs.append((p1, p2)); labels.append(False)
            else:
                raise ValueError(f"Invalid pair line: {parts}")
    return pairs, np.array(labels, dtype=bool)

def threshold_search(sims, labels, num_steps=10000):
    best_thresh, best_acc = None, 0.0
    lo, hi = sims.min(), sims.max()
    for t in np.linspace(lo, hi, num_steps+1):
        preds = sims > t
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc, best_thresh = acc, t
    return best_thresh

def threshold_search_f1(sims, labels, num_steps=10000):
    """Find the threshold that maximizes F1 score on the positive class."""
    best_thresh, best_f1 = None, 0.0
    # Ensure there are positive samples to calculate F1
    if not np.any(labels):
        return sims.min() # Return a default threshold if no positives

    lo, hi = sims.min(), sims.max()
    for t in np.linspace(lo, hi, num_steps+1):
        preds = sims > t
        # Calculate F1 for the positive class (label=True)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0, labels=[True]
        )
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    # If no threshold gives F1 > 0, return the one maximizing accuracy as fallback
    if best_thresh is None:
        return threshold_search(sims, labels, num_steps)
    return best_thresh

def evaluate_sims(sims, labels, n_splits=10):
    mask = ~np.isnan(sims)
    sims, labels = sims[mask], labels[mask]
    kf = KFold(n_splits=n_splits, shuffle=False)
    accs = []
    for tr, te in kf.split(sims):
        thr = threshold_search(sims[tr], labels[tr])
        accs.append(np.mean((sims[te] > thr) == labels[te]))
    return np.mean(accs), np.std(accs)

def evaluate_f1_score(sims, labels, n_splits=10):
    """Calculates 10-fold cross-validated F1 score, optimizing threshold on train folds."""
    mask = ~np.isnan(sims)
    sims, labels = sims[mask], labels[mask]
    if len(sims) == 0:
        return 0.0, 0.0 # Avoid division by zero if no valid pairs

    kf = KFold(n_splits=n_splits, shuffle=False)
    f1_scores = []

    for tr_idx, te_idx in kf.split(sims):
        sims_tr, labels_tr = sims[tr_idx], labels[tr_idx]
        sims_te, labels_te = sims[te_idx], labels[te_idx]

        if len(sims_tr) == 0 or len(sims_te) == 0:
            continue # Skip fold if empty train or test set

        # Find best threshold on the training fold
        best_thresh = threshold_search_f1(sims_tr, labels_tr)

        # Predict on the test fold
        preds_te = sims_te > best_thresh

        # Calculate precision, recall, F1 for this fold (only for the positive class)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_te, preds_te, average='binary', zero_division=0
        )
        f1_scores.append(f1)

    mean_f1 = np.mean(f1_scores) if f1_scores else 0.0
    std_f1 = np.std(f1_scores) if f1_scores else 0.0
    return mean_f1, std_f1

def get_buffalo_embedding(app, image_path):
    """Load image, get face embedding using insightface app."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            # print(f"Warning: Could not read image {image_path}")
            return None
        faces = app.get(img)
        if not faces:
            # print(f"Warning: No face detected in {image_path}")
            return None
        # Use the first detected face
        embedding = faces[0].embedding
        # Return the raw embedding directly to check if it's pre-normalized
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def compute_facenet_embeddings(lfw_dir, mtcnn_fn, model_fn, device, batch_size, workers):
    cropped_dir = f"{lfw_dir}_cropped"
    # Phase 1: Crop all images to disk if not already
    if not os.path.isdir(cropped_dir):
        os.makedirs(cropped_dir, exist_ok=True)
        orig_ds = datasets.ImageFolder(lfw_dir, transform=None)
        orig_ds.samples = [(p, p) for p, _ in orig_ds.samples]
        loader = DataLoader(orig_ds, batch_size=batch_size, num_workers=workers, collate_fn=training.collate_pil)
        print("Cropping images to disk...")
        for x, paths in loader:
            save_paths = [p.replace(lfw_dir, cropped_dir) for p in paths]
            # Create parent directories for cropped images
            for sp in save_paths:
                os.makedirs(os.path.dirname(sp), exist_ok=True)
            mtcnn_fn(x, save_path=save_paths)
    # Phase 2: Load cropped images and extract embeddings
    trans = transforms.Compose([np.float32, transforms.ToTensor(), fixed_image_standardization])
    cropped_ds = datasets.ImageFolder(cropped_dir, transform=trans)
    cropped_ds.samples = [(p, p) for p, _ in cropped_ds.samples]
    embed_loader = DataLoader(cropped_ds, batch_size=batch_size, num_workers=workers, sampler=SequentialSampler(cropped_ds))
    embeddings = {}
    model_fn.eval()
    with torch.no_grad():
        print("Extracting Facenet embeddings...")
        for imgs, paths in embed_loader:
            imgs = imgs.to(device)
            embs = model_fn(imgs).cpu()
            for p_cropped, e in zip(paths, embs):
                # Map cropped path back to original LFW path
                orig_p = p_cropped.replace(cropped_dir, lfw_dir)
                embeddings[orig_p] = e
    return embeddings

def main():
    import argparse
    p = argparse.ArgumentParser("LFW evaluation for Facenet, EdgeFace, and Cascade")
    p.add_argument("--lfw_dir", default="data/lfw/lfw",
                   help="root folder of lfw/<person>/<image>.jpg")
    p.add_argument("--pairs", default="data/lfw/pairs.txt",
                   help="path to pairs.txt (6000 lines, 10 folds)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                   help="torch device")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="band margin around decision boundary for cascade")
    args = p.parse_args()

    # Batch settings matching notebook
    batch_size = 16
    workers = 0 if os.name == 'nt' else 8

    device = torch.device(args.device)
    print(f"Using device: {device}")

    pairs, labels = load_pairs(args.pairs, args.lfw_dir)
    # --- Facenet two-phase pipeline ---
    print("Evaluating Facenet (two-phase)...")
    mtcnn_fn = MTCNN(image_size=160, margin=14, keep_all=False, selection_method='center_weighted_size', post_process=True, device=device)
    model_fn = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    embeddings_fn = compute_facenet_embeddings(args.lfw_dir, mtcnn_fn, model_fn, device, batch_size, workers)
    sims_fn = []
    for p1, p2 in pairs:
        e1, e2 = embeddings_fn.get(p1), embeddings_fn.get(p2)
        sims_fn.append(np.nan if (e1 is None or e2 is None)
                       else F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item())
    sims_fn = np.array(sims_fn, dtype=np.float32)
    acc_fn, std_fn = evaluate_sims(sims_fn, labels)
    print(f"Facenet only: {acc_fn*100:.2f}% ± {std_fn*100:.2f}%")

    # Initialize InsightFace application
    print("Initializing InsightFace...")
    # Use buffalo_s, specify CUDAExecutionProvider for GPU
    buffalo_app = insightface.app.FaceAnalysis(name='buffalo_s', 
                                             root='./insightface_models', # cache directory
                                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    buffalo_app.prepare(ctx_id=0, det_size=(640, 640)) # Use GPU 0
    print("InsightFace initialized.")

    # --- InsightFace buffalo_s pipeline ---
    print("Evaluating InsightFace buffalo_s...")
    embeddings_buffalo = {}
    sims_buffalo = []
    for i, (p1, p2) in enumerate(pairs):
        if i % 100 == 0:
            print(f"\rProcessing pair {i+1}/{len(pairs)}...", end="")
        # Get embeddings directly using the app
        if p1 not in embeddings_buffalo:
            embeddings_buffalo[p1] = get_buffalo_embedding(buffalo_app, p1)
        if p2 not in embeddings_buffalo:
            embeddings_buffalo[p2] = get_buffalo_embedding(buffalo_app, p2)

        e1 = embeddings_buffalo[p1]
        e2 = embeddings_buffalo[p2]

        if e1 is None or e2 is None:
            continue
        else:
            # Cosine similarity for numpy arrays
            similarity = np.dot(e1, e2)
            sims_buffalo.append(similarity)
    print("\nDone processing pairs.")
    sims_buffalo = np.array(sims_buffalo, dtype=np.float32)
    acc_buffalo, std_buffalo = evaluate_sims(sims_buffalo, labels)
    print(f"InsightFace buffalo_s only: {acc_buffalo*100:.2f}% ± {std_buffalo*100:.2f}%")

    # --- Cascading ---
    print("Evaluating Cascade...")
    # Use buffalo_s similarities for cascade
    valid_mask = ~np.isnan(sims_buffalo) & ~np.isnan(sims_fn)
    valid_indices = np.arange(len(valid_mask))[valid_mask]
    valid_labels = labels[valid_mask]
    valid_sims_buffalo = sims_buffalo[valid_mask]
    valid_sims_fn = sims_fn[valid_mask]

    # calibration on first 9 folds (test sets of folds 0..8)
    kf = KFold(n_splits=10, shuffle=False)
    calib_idx = np.concatenate([t for i, (_, t) in enumerate(kf.split(valid_sims_buffalo)) if i < 9])
    sims_c = valid_sims_buffalo.copy() # Start with buffalo similarities
    mu_same = valid_sims_buffalo[calib_idx][valid_labels[calib_idx]].mean()
    mu_diff = valid_sims_buffalo[calib_idx][~valid_labels[calib_idx]].mean()
    band_low = mu_diff + args.alpha
    band_high = mu_same - args.alpha
    ambiguous = (valid_sims_buffalo > band_low) & (valid_sims_buffalo < band_high)
    use_fn = np.where(ambiguous)[0]
    # Overwrite ambiguous cases with Facenet similarities
    sims_c[use_fn] = valid_sims_fn[use_fn]
    acc_c, std_c = evaluate_sims(sims_c, valid_labels)
    pct = len(use_fn)/len(valid_sims_buffalo)*100
    print(f"Cascade (buffalo_s -> Facenet, alpha={args.alpha}): {acc_c*100:.2f}% ± {std_c*100:.2f}%, "
          f"Facenet calls: {len(use_fn)}/{len(valid_sims_buffalo)} ({pct:.1f}%)")

    # --- Base Rate Experiments --- (using buffalo_s cascade)
    print("\n--- Base Rate Experiments (Cascade F1 Score) ---")
    target_base_rates = [0.01, 0.05, 0.10, 0.50]
    original_indices = np.arange(len(labels))
    same_indices = original_indices[labels]
    diff_indices = original_indices[~labels]

    # Ensure we use the valid indices where both models produced embeddings
    valid_mask = ~np.isnan(sims_buffalo) & ~np.isnan(sims_fn)
    valid_indices = np.arange(len(valid_mask))[valid_mask]
    valid_labels = labels[valid_mask]
    valid_sims_buffalo = sims_buffalo[valid_mask]
    valid_sims_fn = sims_fn[valid_mask]

    valid_same_indices = valid_indices[valid_labels]
    valid_diff_indices = valid_indices[~valid_labels]

    np.random.seed(42) # for reproducible sampling

    for rate in target_base_rates:
        num_diff = len(valid_diff_indices)
        num_same_needed = int(round(num_diff * rate / (1.0 - rate)))

        if num_same_needed > len(valid_same_indices):
            print(f"Warning: Not enough 'same' pairs for base rate {rate*100}%. Using all {len(valid_same_indices)}.")
            sampled_same_indices = valid_same_indices
        elif num_same_needed <= 0:
            print(f"Skipping base rate {rate*100}% as it requires 0 or fewer 'same' pairs.")
            continue
        else:
            sampled_same_indices = np.random.choice(valid_same_indices, num_same_needed, replace=False)

        resampled_indices = np.concatenate([sampled_same_indices, valid_diff_indices])
        resampled_labels = labels[resampled_indices]
        resampled_sims_buffalo = sims_buffalo[resampled_indices]
        resampled_sims_fn = sims_fn[resampled_indices]

        # Apply the same cascade logic determined earlier (fixed alpha band)
        resampled_sims_c = resampled_sims_buffalo.copy()
        ambiguous_resampled = (resampled_sims_buffalo > band_low) & (resampled_sims_buffalo < band_high)
        use_fn_resampled = np.where(ambiguous_resampled)[0]
        resampled_sims_c[use_fn_resampled] = resampled_sims_fn[use_fn_resampled]

        # Evaluate F1 score using 10-fold CV on the resampled data
        f1_c, std_f1_c = evaluate_f1_score(resampled_sims_c, resampled_labels)

        actual_rate = np.mean(resampled_labels)
        print(f"Base Rate: {actual_rate*100:.2f}% ({len(sampled_same_indices)}/{len(resampled_indices)}) -> Cascade F1: {f1_c:.4f} ± {std_f1_c:.4f}")

if __name__ == '__main__':
    main()

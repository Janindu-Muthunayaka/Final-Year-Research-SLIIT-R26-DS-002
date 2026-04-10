# =============================================================================
# trainer.py
# Topological Sinhala Character Recognition via Medial Axis Transform
# Model: EfficientNetV2-S | Classes: 759 | Input: 512x512 Grayscale Skeletons
# Environment: Python 3.11.9 | PyTorch 2.5.1+cu121 | RTX 4050 Laptop GPU
# =============================================================================

import os
import sys
import json
import time
import copy
import shutil
import signal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast

from torchvision import datasets, transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

import psutil

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_ROOT     = r"E:\Sliit\Research\Repositoryv2\Datasets\SkeletonImages"
MODEL_SAVE_DIR   = r"E:\Sliit\Research\Repositoryv2\Final-Year-Research-SLIIT-R26-DS-002\2-Model"
REPORT_DIR       = r"E:\Sliit\Research\Repositoryv2\Final-Year-Research-SLIIT-R26-DS-002\2-Model"
CHECKPOINT_PATH  = os.path.join(MODEL_SAVE_DIR, "checkpoint.pth")
CHECKPOINT_TMP   = os.path.join(MODEL_SAVE_DIR, "checkpoint.tmp.pth")

# EfficientNetV2-S is fed 512x512 intentionally to match native skeleton size.
# Its native training resolution is 384x384; the upsized input is deliberate
# and provides richer spatial detail for sparse stroke data.
IMAGE_SIZE          = 384
BATCH_SIZE          = 16
NUM_WORKERS         = 4
LEARNING_RATE       = 1e-4       # Low LR to protect ImageNet pretrained weights
MAX_EPOCHS          = 20
EARLY_STOP_PATIENCE = 5
VAL_SPLIT           = 0.2
RANDOM_SEED         = 42

# =============================================================================
# FOLDER NAME → LOGICAL CLASS REMAPPING
#
# Windows forbids '!' and '/' in folder names, so those two punctuation
# classes use safe Windows-legal names on disk. This map translates the
# on-disk folder name back to the canonical class label so that
# class_mapping.json stays consistent with the rest of the pipeline.
#
# All other class folders (including xpunct{0758 and xpunct}_0759 which use
# '{' and '}' — characters Windows does allow) are used as-is.
# =============================================================================

FOLDER_REMAP = {
    "xpunct_question_0756": "xpunct_?_0756",   # '?' is legal in folder names
                                                # but kept as named folder for clarity
    "xpunct_slash_0757":    "xpunct_/_0757",   # '/' is NOT legal on Windows — named folder
    # Note: xpunct_!0755 uses '!' which IS legal on Windows, so no remap needed.
    # If your system saved it differently (e.g. "xpunct_exclaim_0755"), add it here.
}

# =============================================================================
# STOP FLAG — mutable list so signal handler can modify it from any scope
# =============================================================================

_stop = [False]

def _handle_sigint(sig, frame):
    if not _stop[0]:
        print("\n\n  [!] Ctrl+C detected — finishing batch, then saving & exiting cleanly...")
        print("      (Press Ctrl+C again to force-quit WITHOUT saving)\n")
        _stop[0] = True
    else:
        print("\n  [!] Force quit. Cleaning temp files...")
        if os.path.exists(CHECKPOINT_TMP):
            os.remove(CHECKPOINT_TMP)
        sys.exit(1)

signal.signal(signal.SIGINT, _handle_sigint)

# =============================================================================
# SAFE IMAGE LOADER
# =============================================================================

def safe_loader(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(255, 255, 255))

# =============================================================================
# TRANSFORMS
# Skeleton images are grayscale → 3-channel for EfficientNetV2-S compatibility.
# No horizontal flip — Sinhala characters are NOT mirror-symmetric.
# Resize to 384x384 to match EfficientNetV2-S native resolution.
# =============================================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =============================================================================
# HELPERS
# =============================================================================

def format_eta(seconds):
    if seconds < 0 or seconds != seconds:
        return "calculating..."
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def get_hw_stats(device):
    stats = []
    if device.type == "cuda":
        vram_used  = torch.cuda.memory_allocated(0) / 1024**2
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        stats.append(f"VRAM: {vram_used:.0f}/{vram_total:.0f}MB")
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            temp = result.stdout.strip()
            if temp:
                stats.append(f"GPU Temp: {temp}°C")
        except Exception:
            pass
    cpu_pct      = psutil.cpu_percent(interval=None)
    ram          = psutil.virtual_memory()
    ram_used_gb  = ram.used  / 1024**3
    ram_total_gb = ram.total / 1024**3
    stats.append(f"CPU: {cpu_pct:.0f}%")
    stats.append(f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB")
    return "  |  ".join(stats)


def save_checkpoint(path, tmp_path, epoch, model, optimizer, scheduler, scaler,
                    history, best_val_loss, best_epoch, patience_counter,
                    num_classes, is_best, model_save_dir):
    state = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "history":              history,
        "best_val_loss":        best_val_loss,
        "best_epoch":           best_epoch,
        "patience_counter":     patience_counter,
        "num_classes":          num_classes,
    }
    torch.save(state, tmp_path)
    shutil.move(tmp_path, path)
    if is_best:
        best_path = os.path.join(model_save_dir, "best_model.pth")
        torch.save(state, best_path)


def cleanup_old_checkpoints(checkpoint_path, tmp_path, model_save_dir):
    """Always start fresh — delete any old checkpoint files."""
    removed = []
    for fpath in [checkpoint_path, tmp_path,
                  os.path.join(model_save_dir, "best_model.pth"),
                  os.path.join(model_save_dir, "final_model.pth")]:
        if os.path.exists(fpath):
            os.remove(fpath)
            removed.append(os.path.basename(fpath))
    if removed:
        print(f"  [Fresh Start] Removed old checkpoint files: {', '.join(removed)}")
    else:
        print("  [Fresh Start] No old checkpoints found — clean slate.")


# =============================================================================
# MAIN — required on Windows to prevent multiprocessing spawn errors
# =============================================================================

if __name__ == '__main__':

    # Directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  Sinhala Skeleton Trainer — EfficientNetV2-S")
    print(f"{'='*65}")
    print(f"  Device   : {device}")
    if device.type == "cuda":
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  VRAM     : {total_vram:.0f} MB total")
    print(f"  Dataset  : {DATASET_ROOT}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # ALWAYS START FRESH — delete any old checkpoints
    # ------------------------------------------------------------------
    cleanup_old_checkpoints(CHECKPOINT_PATH, CHECKPOINT_TMP, MODEL_SAVE_DIR)
    print()

    # ------------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------------
    print("  Loading dataset structure...")
    full_dataset = datasets.ImageFolder(root=DATASET_ROOT, loader=safe_loader)

    num_classes = len(full_dataset.classes)
    total_imgs  = len(full_dataset)
    print(f"  Classes found  : {num_classes}")
    print(f"  Total images   : {total_imgs}\n")

    # Save class mapping — apply FOLDER_REMAP so downstream tools see logical names
    raw_class_to_idx = full_dataset.class_to_idx           # folder_name → int
    logical_class_mapping = {
        str(idx): FOLDER_REMAP.get(folder, folder)
        for folder, idx in raw_class_to_idx.items()
    }
    mapping_path = os.path.join(REPORT_DIR, "class_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(logical_class_mapping, f, ensure_ascii=False, indent=2)
    print(f"  class_mapping.json → {mapping_path}\n")

    # ------------------------------------------------------------------
    # TRAIN / VAL SPLIT — stratified 80/20
    # ------------------------------------------------------------------
    targets = full_dataset.targets
    train_idx, val_idx = train_test_split(
        list(range(total_imgs)),
        test_size=VAL_SPLIT,
        stratify=targets,
        random_state=RANDOM_SEED
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform   = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    print(f"  Train samples  : {len(train_dataset)}")
    print(f"  Val samples    : {len(val_dataset)}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Train batches  : {len(train_loader)}\n")

    # ------------------------------------------------------------------
    # MODEL — EfficientNetV2-S
    # ------------------------------------------------------------------
    print("  Building EfficientNetV2-S model...")
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # Swap classifier head: 1280 → 759 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Output layer   : {in_features} → {num_classes} classes")
    print(f"  Total params   : {total_params / 1e6:.1f}M")
    print(f"  Input size     : {IMAGE_SIZE}x{IMAGE_SIZE}  "
          f"(matches EfficientNetV2-S native resolution)")
    print(f"  torch.compile(): Disabled (not supported on Windows without MSVC)\n")

    # ------------------------------------------------------------------
    # LOSS / OPTIMIZER / SCHEDULER / SCALER
    # ------------------------------------------------------------------
    # Plain CrossEntropyLoss — training data is balanced (320 images/class),
    # so frequency weighting would only add noise with no benefit.
    criterion = nn.CrossEntropyLoss()

    # AdamW with conservative LR — protects pretrained ImageNet weights on
    # first contact with sparse skeleton data
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # CosineAnnealingLR: T_max = MAX_EPOCHS so the LR anneals once across the
    # full training budget, keeping V2-S convergence smooth over its longer run.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    scaler = GradScaler(device="cuda" if device.type == "cuda" else "cpu")

    # ------------------------------------------------------------------
    # INIT TRAINING STATE — always fresh, no resume
    # ------------------------------------------------------------------
    start_epoch      = 1
    history          = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_loss    = float("inf")
    best_model_wts   = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_epoch       = 0

    print("  Starting fresh — no checkpoint resume.\n")

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    print(f"{'='*65}")
    print(f"  Model     : EfficientNetV2-S (~{total_params/1e6:.0f}M params)")
    print(f"  Training  : Epochs {start_epoch} → {MAX_EPOCHS}")
    print(f"  Optimizer : AdamW  lr={LEARNING_RATE}  wd=1e-4")
    print(f"  Scheduler : CosineAnnealingLR  T_max={MAX_EPOCHS}  eta_min=1e-6")
    print(f"  Loss      : CrossEntropyLoss (unweighted — balanced dataset)")
    print(f"  Patience  : {EARLY_STOP_PATIENCE} epochs (early stopping)")
    print(f"  AMP       : torch.amp.autocast  (critical for 6GB VRAM)")
    print(f"  Ctrl+C    : Safe stop — saves checkpoint, then exits cleanly")
    print(f"{'='*65}\n")

    global_start  = time.time()
    epoch_times   = []
    stopped_early = False

    for epoch in range(start_epoch, MAX_EPOCHS + 1):

        if _stop[0]:
            print("  [Stop] Safe stop before epoch started.")
            break

        epoch_start = time.time()
        current_lr  = scheduler.get_last_lr()[0] if epoch > 1 else LEARNING_RATE

        # TRAIN PHASE
        model.train()
        running_loss, running_correct = 0.0, 0
        batch_start = time.time()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:02d}/{MAX_EPOCHS} [Train]",
                    leave=False, dynamic_ncols=True)

        for batch_idx, (images, labels) in enumerate(pbar):

            if _stop[0]:
                pbar.close()
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss    += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()

            elapsed        = batch_idx + 1
            time_per_batch = (time.time() - batch_start) / elapsed
            batch_eta      = time_per_batch * (len(train_loader) - elapsed)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "eta": format_eta(batch_eta)})

        # Step scheduler once per epoch (after optimizer step)
        scheduler.step()

        train_loss = running_loss    / len(train_dataset)
        train_acc  = running_correct / len(train_dataset)

        # VALIDATION PHASE
        model.eval()
        val_loss_sum, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader,
                                       desc=f"Epoch {epoch:02d}/{MAX_EPOCHS} [Val]  ",
                                       leave=False, dynamic_ncols=True):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                val_correct  += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_loss_sum / len(val_dataset)
        val_acc  = val_correct  / len(val_dataset)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        avg_epoch_time = sum(epoch_times[-5:]) / len(epoch_times[-5:])
        overall_eta    = avg_epoch_time * (MAX_EPOCHS - epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss    = val_loss
            best_model_wts   = copy.deepcopy(model.state_dict())
            best_epoch       = epoch
            patience_counter = 0
            ckpt_status      = f"✓ New best val_loss={val_loss:.4f} — best_model.pth updated"
        else:
            patience_counter += 1
            ckpt_status      = f"✗ No improvement  |  Patience: {patience_counter}/{EARLY_STOP_PATIENCE}"

        hw = get_hw_stats(device)

        print(f"\n  {'─'*61}")
        print(f"  Epoch     : {epoch:02d}/{MAX_EPOCHS}  |  Time: {epoch_time:.1f}s  |  Overall ETA: {format_eta(overall_eta)}")
        print(f"  LR        : {current_lr:.2e}")
        print(f"  Train     : Loss={train_loss:.4f}  Acc={train_acc*100:.2f}%")
        print(f"  Val       : Loss={val_loss:.4f}  Acc={val_acc*100:.2f}%")
        print(f"  Hardware  : {hw}")
        print(f"  Checkpoint: {ckpt_status}")
        print(f"  {'─'*61}")

        save_checkpoint(
            CHECKPOINT_PATH, CHECKPOINT_TMP, epoch, model, optimizer, scheduler, scaler,
            history, best_val_loss, best_epoch, patience_counter,
            num_classes, is_best, MODEL_SAVE_DIR
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  [Early Stop] Triggered at epoch {epoch}.")
            print(f"  [Early Stop] Best epoch: {best_epoch}  |  Best val_loss: {best_val_loss:.4f}\n")
            stopped_early = True
            break

        if _stop[0]:
            print(f"\n  [Stop] Paused after epoch {epoch}. Re-run trainer.py to resume.\n")
            break

    # ------------------------------------------------------------------
    # FINALISE — restore best weights, save final model
    # ------------------------------------------------------------------
    total_time = time.time() - global_start

    print(f"\n{'='*65}")
    print(f"  {'Stopped' if _stop[0] else 'Training Complete'}")
    print(f"{'='*65}")

    model.load_state_dict(best_model_wts)
    final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model.pth")
    torch.save({
        "epoch":            best_epoch,
        "model_state_dict": model.state_dict(),
        "num_classes":      num_classes,
        "val_loss":         best_val_loss,
        "val_acc":          history["val_acc"][best_epoch - 1] if best_epoch > 0 else 0.0,
        "architecture":     "efficientnetv2_s",
    }, final_model_path)
    print(f"  Final model   → {final_model_path}")

    # PLOTS
    if len(history["train_loss"]) > 0:
        epochs_ran = list(range(1, len(history["train_loss"]) + 1))
        fig, axes  = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle("EfficientNetV2-S — Sinhala Skeleton Training", fontsize=14, fontweight="bold")

        # Loss
        axes[0].plot(epochs_ran, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
        axes[0].plot(epochs_ran, history["val_loss"],   label="Val Loss",   color="#ff7f0e", linewidth=2, linestyle="--")
        if best_epoch > 0:
            axes[0].axvline(best_epoch, color="green", linestyle=":", linewidth=1.5, label=f"Best (ep {best_epoch})")
        axes[0].set_title("Loss per Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs_ran, [a*100 for a in history["train_acc"]], label="Train Acc", color="#1f77b4", linewidth=2)
        axes[1].plot(epochs_ran, [a*100 for a in history["val_acc"]],   label="Val Acc",   color="#ff7f0e", linewidth=2, linestyle="--")
        if best_epoch > 0:
            axes[1].axvline(best_epoch, color="green", linestyle=":", linewidth=1.5, label=f"Best (ep {best_epoch})")
        axes[1].set_title("Accuracy per Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # LR Schedule
        axes[2].plot(epochs_ran, history["lr"], color="#9467bd", linewidth=2)
        axes[2].set_title("Learning Rate Schedule (CosineAnnealing)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(REPORT_DIR, "training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Training curves → {plot_path}")

    # SUMMARY JSON
    summary = {
        "model":            "EfficientNetV2-S",
        "num_classes":      num_classes,
        "image_size":       IMAGE_SIZE,
        "batch_size":       BATCH_SIZE,
        "total_images":     total_imgs,
        "train_images":     len(train_dataset),
        "val_images":       len(val_dataset),
        "best_epoch":       best_epoch,
        "best_val_loss":    round(best_val_loss, 6),
        "best_val_acc_pct": round(history["val_acc"][best_epoch - 1] * 100, 2) if best_epoch > 0 else 0.0,
        "epochs_ran":       len(history["train_loss"]),
        "early_stopped":    stopped_early,
        "manually_stopped": bool(_stop[0]),
        "total_time_min":   round(total_time / 60, 2),
        "optimizer":        "AdamW",
        "scheduler":        f"CosineAnnealingLR(T_max={MAX_EPOCHS}, eta_min=1e-6)",
        "loss":             "CrossEntropyLoss (unweighted)",
        "device":           str(device),
        "gpu":              torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch_version":    torch.__version__,
    }

    summary_path = os.path.join(REPORT_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  Summary       → {summary_path}")
    print(f"\n  Best Epoch    : {summary['best_epoch']}")
    print(f"  Best Val Loss : {summary['best_val_loss']}")
    print(f"  Best Val Acc  : {summary['best_val_acc_pct']}%")
    print(f"  Total Time    : {summary['total_time_min']} minutes")
    print(f"{'='*65}\n")
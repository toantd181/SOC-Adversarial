"""
src/defense/adv_train.py

Adversarial Training Pipeline – Robust CNN (TrafficSignNet)
===========================================================

The AI Vaccine Analogy
----------------------
Standard training teaches a model to classify "healthy" inputs — images the
world naturally produces.  Adversarial training is the immunisation shot: we
deliberately inject weakened versions of the threat (PGD adversarial examples)
into every training batch, forcing the model's decision boundaries to harden
against them.

Just as a vaccine exposes the immune system to an attenuated pathogen so it
learns to recognise and defeat the real virus, adversarial training exposes
the model to worst-case perturbations during learning so that, at inference
time, the real attack can no longer shift the model's prediction.

The cost, like any vaccine, is a slight reduction in "clean" peak performance
(natural accuracy typically drops 1–5%) in exchange for dramatically improved
robustness — the model that previously collapsed to near-zero accuracy under
PGD now maintains meaningful performance against the adversary.

PGD (Projected Gradient Descent) – Madry et al., 2018
------------------------------------------------------
PGD is the strong, iterative adversary used both to *attack* in Red Team
evaluation and to *train* in Blue Team defence.  At each of ``steps``
iterations it takes a step of size ``alpha`` in the direction that maximises
the cross-entropy loss, then projects the cumulative perturbation back onto
the L∞ ball of radius ``eps`` around the original input.  Using PGD during
training (rather than the cheaper FGSM) yields a model that generalises its
robustness to a much wider family of attacks.

Training strategy: 50 / 50 mixed-batch
---------------------------------------
Each mini-batch is split: half the samples are trained on their **clean**
versions, half on their **PGD-adversarial** versions.  The two halves are
concatenated before the forward pass so a single backward step covers both
distributions.  This is empirically more stable than pure adversarial
training because:
  * The clean gradient signal prevents catastrophic drift of the feature
    representations, preserving natural accuracy.
  * The adversarial gradient simultaneously flattens the loss landscape
    around adversarial inputs, building robustness.
  * Compute cost is exactly one PGD attack per batch (on half the images),
    making it feasible on Kaggle T4 / P100 GPUs within reasonable time.

Dual Validation
---------------
Saving the best checkpoint purely on clean validation accuracy would be
misleading — a model can achieve high clean accuracy while remaining brittle.
We track both clean and PGD robust validation accuracy and save the checkpoint
whenever *robust* accuracy improves.  This ensures ``weights/robust_cnn.pth``
is the genuinely most attack-resistant model seen during training.

Usage (Kaggle / Cloud GPU):
    python -m src.defense.adv_train \
        --data_dir  data/processed \
        --epochs    20 \
        --batch_size 64 \
        --lr        1e-3 \
        --eps       0.03137 \
        --alpha     0.00784 \
        --steps     7
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")           # headless – safe for Kaggle / SSH
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchattacks
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.api.logger import get_logger
from src.data.dataset import get_dataloaders
from src.models.cnn_classifier import TrafficSignNet

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SOC colour palette (consistent with Red Team visualisations)
# ---------------------------------------------------------------------------

SOC_BG      = "#0f0f14"
SOC_SURFACE = "#1a1a24"
SOC_GRID    = "#2a2a3a"
SOC_TEXT    = "#e0e0e0"
SOC_SUBTEXT = "#9e9e9e"
SOC_BLUE    = "#2196F3"   # adversarial train loss
SOC_GREEN   = "#4CAF50"   # clean val accuracy
SOC_ORANGE  = "#FF9800"   # robust val accuracy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for the adversarial training run.

    PGD hyperparameters
    -------------------
    eps   – L∞ perturbation budget.  8/255 ≈ 0.0314 is the community standard
            for GTSRB / CIFAR-scale benchmarks.
    alpha – Per-step size.  Madry et al. recommend alpha = eps / 4; at 7 steps
            this gives the attack enough range to explore the ε-ball without
            overshooting.
    steps – Number of PGD iterations.  PGD-7 is the canonical training attack;
            PGD-20 or higher is used for evaluation to stress-test the defence.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial Training of the AI-SOC Robust CNN (PGD-AT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the processed dataset (train/ and val/ splits).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size.  Each batch is split 50/50 clean vs adversarial.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for the Adam optimiser.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=8 / 255,
        help="PGD L∞ perturbation budget ε (e.g. 8/255 ≈ 0.0314).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2 / 255,
        help="PGD per-step size α (e.g. 2/255 ≈ 0.0078).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=7,
        help="Number of PGD iterations per batch (PGD-7 is the training standard).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=43,
        help="Number of output classes (43 for GTSRB).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience: epochs without robust_val_acc improvement.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="weights",
        help="Directory where model checkpoints are saved.",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Directory where training plots are saved.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be ≥ 1.")
    if args.batch_size < 2:
        parser.error("--batch_size must be ≥ 2 (50/50 split requires at least 1 sample per half).")
    if args.lr <= 0:
        parser.error("--lr must be a positive float.")
    if not (0.0 < args.eps <= 1.0):
        parser.error("--eps must be in (0, 1].")
    if not (0.0 < args.alpha <= args.eps):
        parser.error("--alpha must be in (0, eps].")
    if args.steps < 1:
        parser.error("--steps must be ≥ 1.")

    return args


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Halt training when ``robust_val_acc`` fails to improve for ``patience``
    consecutive epochs.

    Robust accuracy (not clean accuracy) is the monitored signal because
    the goal of adversarial training is to maximise robustness; a model that
    trades clean accuracy for robustness should not be stopped prematurely.

    Parameters
    ----------
    patience  : Epochs to wait after the last improvement.
    min_delta : Minimum absolute gain (pp) that qualifies as an improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.1) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter:   int         = 0
        self.best_score: float | None = None
        self.triggered: bool        = False

    def step(self, metric: float) -> bool:
        """
        Update state.

        Returns
        -------
        bool
            True → stop training; False → continue.
        """
        if self.best_score is None or metric > self.best_score + self.min_delta:
            self.best_score = metric
            self.counter    = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping: %d / %d  (best robust_val_acc=%.2f%%)",
                self.counter, self.patience, self.best_score,
            )
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False


# ---------------------------------------------------------------------------
# Core training pass
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    pgd:       torchattacks.PGD,
    device:    torch.device,
    epoch:     int,
    total_epochs: int,
) -> float:
    """
    Execute one adversarial training epoch using a 50 / 50 mixed batch strategy.

    For every mini-batch the function:
      1. Splits the batch in half (first half → clean, second half → PGD adversarial).
      2. Generates PGD adversarial examples for the second half **in-place**
         within the training loop (on-the-fly attack generation).
      3. Concatenates both halves and performs a single forward / backward pass.

    This mixed strategy stabilises training compared to pure adversarial batches
    because the clean half anchors the feature representations and prevents the
    catastrophic forgetting of natural image statistics that can occur when the
    model sees only worst-case inputs.

    Parameters
    ----------
    model        : Network being trained (set to train mode internally).
    loader       : Training DataLoader.
    criterion    : Cross-entropy loss.
    optimiser    : Adam optimiser.
    pgd          : Instantiated ``torchattacks.PGD`` attack object.
    device       : Target device.
    epoch        : Current epoch index (1-based) for display.
    total_epochs : Maximum epochs for the tqdm description.

    Returns
    -------
    float
        Mean adversarial training loss over the epoch.
    """
    model.train()
    running_loss: float = 0.0
    n_batches:    int   = len(loader)

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:>3}/{total_epochs} [ADV-TRAIN]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_size = images.size(0)

        # ---- 50 / 50 split -----------------------------------------------
        # Ceiling division keeps the split balanced even for odd batch sizes.
        n_clean = batch_size // 2
        n_adv   = batch_size - n_clean     # picks up the extra sample if odd

        clean_images = images[:n_clean]
        clean_labels = labels[:n_clean]
        adv_src_imgs = images[n_clean:]    # source images for attack generation
        adv_labels   = labels[n_clean:]

        # ---- On-the-fly PGD adversarial example generation ---------------
        # model.eval() is called internally by torchattacks; we restore
        # model.train() immediately after.  The PGD object holds a reference
        # to the model so no extra arguments are needed here.
        adv_images: torch.Tensor = pgd(adv_src_imgs, adv_labels)
        model.train()                      # restore training mode after attack

        # ---- Mixed forward pass ------------------------------------------
        mixed_images = torch.cat([clean_images, adv_images], dim=0)
        mixed_labels = torch.cat([clean_labels, adv_labels], dim=0)

        optimiser.zero_grad(set_to_none=True)
        logits: torch.Tensor = model(mixed_images)
        loss:   torch.Tensor = criterion(logits, mixed_labels)
        loss.backward()
        optimiser.step()

        batch_loss = loss.item()
        running_loss += batch_loss
        progress.set_postfix(adv_loss=f"{batch_loss:.4f}")

    return running_loss / n_batches


# ---------------------------------------------------------------------------
# Dual validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_accuracy(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute accuracy (%) over ``loader`` without modifying gradients."""
    correct = 0
    total   = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds  = model(images).argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total   += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    pgd_val:   torchattacks.PGD,
    device:    torch.device,
    epoch:     int,
    total_epochs: int,
) -> Tuple[float, float]:
    """
    Evaluate the model on both the clean and PGD-attacked validation sets.

    Two separate passes are required:
      * **Clean pass** – ``@torch.no_grad()`` applied; standard accuracy.
      * **Adversarial pass** – autograd must be active for the PGD attack;
        ``torch.no_grad()`` is applied only around the final forward pass
        for accuracy measurement, not around the attack itself.

    Parameters
    ----------
    model        : Network to evaluate.
    loader       : Validation DataLoader.
    pgd_val      : PGD attack object (may use more steps than the training PGD
                   to provide a harder evaluation signal).
    device       : Target device.
    epoch        : Current epoch (for tqdm display).
    total_epochs : Maximum epochs (for tqdm display).

    Returns
    -------
    Tuple[float, float]
        (clean_val_accuracy_percent, robust_val_accuracy_percent)
    """
    # ---- Pass 1: clean accuracy ------------------------------------------
    model.eval()
    clean_correct = 0
    clean_total   = 0

    for images, labels in tqdm(
        loader,
        desc=f"Epoch {epoch:>3}/{total_epochs} [VAL-CLEAN] ",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        clean_correct += int((preds == labels).sum().item())
        clean_total   += labels.size(0)

    clean_acc = 100.0 * clean_correct / clean_total if clean_total > 0 else 0.0

    # ---- Pass 2: robust accuracy (PGD-attacked) --------------------------
    # torchattacks sets model to eval mode internally; train mode is irrelevant
    # here since we do not update weights during validation.
    robust_correct = 0
    robust_total   = 0

    for images, labels in tqdm(
        loader,
        desc=f"Epoch {epoch:>3}/{total_epochs} [VAL-ROBUST]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        adv_images = pgd_val(images, labels)   # autograd active for attack

        with torch.no_grad():
            preds = model(adv_images).argmax(dim=1)

        robust_correct += int((preds == labels).sum().item())
        robust_total   += labels.size(0)

    robust_acc = 100.0 * robust_correct / robust_total if robust_total > 0 else 0.0

    return clean_acc, robust_acc


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_training_plots(
    adv_train_losses:  List[float],
    clean_val_accs:    List[float],
    robust_val_accs:   List[float],
    best_epoch:        int,
    save_path:         Path,
) -> None:
    """
    Generate and save the adversarial training history figure.

    Three-line plot on a shared epoch x-axis:
      • Blue  – Adversarial training loss (left y-axis, lower is better).
      • Green – Clean validation accuracy % (right y-axis, higher is better).
      • Orange – Robust validation accuracy % (right y-axis, higher is better).

    The gap between the green and orange lines is the **robustness gap** –
    how much accuracy the model sacrifices on clean inputs to gain resilience
    against adversarial inputs.  A well-trained robust model minimises this
    gap while keeping robust accuracy as high as possible.

    A vertical dashed line marks the best checkpoint epoch (highest robust acc).

    Parameters
    ----------
    adv_train_losses : Per-epoch mean adversarial training loss.
    clean_val_accs   : Per-epoch clean validation accuracy (%).
    robust_val_accs  : Per-epoch robust validation accuracy (%).
    best_epoch       : Epoch index (1-based) of the saved best checkpoint.
    save_path        : Output PNG path (parent dirs created if absent).
    """
    epochs_range = range(1, len(adv_train_losses) + 1)

    fig, ax_loss = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(SOC_BG)
    ax_loss.set_facecolor(SOC_SURFACE)

    # Shared grid
    ax_loss.yaxis.grid(True, color=SOC_GRID, linewidth=0.7, linestyle="--", zorder=0)
    ax_loss.set_axisbelow(True)

    # ---- Left axis: adversarial training loss ----------------------------
    loss_line, = ax_loss.plot(
        epochs_range, adv_train_losses,
        label="Adv. Train Loss",
        color=SOC_BLUE, linewidth=2, marker="o", markersize=4, zorder=3,
    )
    ax_loss.set_xlabel("Epoch", fontsize=11, color=SOC_TEXT, labelpad=8)
    ax_loss.set_ylabel("Adversarial Training Loss", fontsize=11,
                        color=SOC_BLUE, labelpad=8)
    ax_loss.tick_params(axis="y", colors=SOC_BLUE, labelsize=9)
    ax_loss.tick_params(axis="x", colors=SOC_TEXT, labelsize=9)

    # ---- Right axis: accuracy lines -------------------------------------
    ax_acc = ax_loss.twinx()
    ax_acc.set_facecolor("none")          # transparent so left axis bg shows

    clean_line, = ax_acc.plot(
        epochs_range, clean_val_accs,
        label="Clean Val Acc (%)",
        color=SOC_GREEN, linewidth=2, marker="s", markersize=4,
        linestyle="-", zorder=3,
    )
    robust_line, = ax_acc.plot(
        epochs_range, robust_val_accs,
        label="Robust Val Acc (%)",
        color=SOC_ORANGE, linewidth=2, marker="^", markersize=4,
        linestyle="-", zorder=3,
    )

    # Shade the robustness gap between clean and robust accuracy
    ax_acc.fill_between(
        epochs_range,
        clean_val_accs,
        robust_val_accs,
        alpha=0.10,
        color=SOC_ORANGE,
        label="Robustness gap",
        zorder=2,
    )

    ax_acc.set_ylabel("Validation Accuracy (%)", fontsize=11,
                       color=SOC_TEXT, labelpad=8)
    ax_acc.tick_params(axis="y", colors=SOC_SUBTEXT, labelsize=9)
    ax_acc.set_ylim(0, 110)

    # ---- Best-epoch marker ----------------------------------------------
    best_robust = robust_val_accs[best_epoch - 1]
    ax_acc.axvline(
        x=best_epoch, color=SOC_ORANGE,
        linestyle="--", linewidth=1.2, alpha=0.7,
        label=f"Best epoch ({best_epoch})",
        zorder=4,
    )
    ax_acc.annotate(
        f"  Best\n  {best_robust:.2f}%",
        xy=(best_epoch, best_robust),
        color=SOC_ORANGE, fontsize=8.5,
    )

    # ---- Spines ---------------------------------------------------------
    for ax in (ax_loss, ax_acc):
        for spine_name, spine in ax.spines.items():
            spine.set_color(SOC_GRID)
            spine.set_linewidth(0.8)

    # ---- Legend ---------------------------------------------------------
    handles = [loss_line, clean_line, robust_line]
    labels_  = [h.get_label() for h in handles]
    ax_loss.legend(
        handles, labels_,
        loc="upper right",
        fontsize=9,
        framealpha=0.25,
        facecolor=SOC_BG,
        edgecolor=SOC_GRID,
        labelcolor=SOC_TEXT,
    )

    # ---- Titles ---------------------------------------------------------
    fig.suptitle(
        "AI-SOC Blue Team — Adversarial Training History (PGD-AT)",
        fontsize=13, fontweight="bold", color=SOC_TEXT, y=0.98,
    )
    ax_loss.set_title(
        "The shaded area shows the robustness gap: clean accuracy vs PGD robust accuracy",
        fontsize=9, color=SOC_SUBTEXT, pad=6,
    )

    # ---- Save -----------------------------------------------------------
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Adversarial training history plot saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_adv_training(args: argparse.Namespace) -> Dict[str, List[float]]:
    """
    Orchestrate the full adversarial training pipeline.

    1. Resolve output directories.
    2. Detect hardware device.
    3. Instantiate model, optimiser, scheduler, criterion, and PGD objects.
    4. Execute the adversarial train / dual-validate loop with early stopping.
    5. Save the best robust checkpoint and the training history plot.

    Parameters
    ----------
    args : Parsed namespace from ``parse_args()``.

    Returns
    -------
    Dict[str, List[float]]
        Training history with keys:
        ``adv_train_loss``, ``clean_val_acc``, ``robust_val_acc``.
    """
    weights_dir = Path(args.weights_dir)
    logs_dir    = Path(args.logs_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    weights_path = weights_dir / "robust_cnn.pth"
    plot_path    = logs_dir    / "adv_training_history.png"

    # ---- Device ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Hardware device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s  |  VRAM: %.2f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
        )

    # ---- Data ------------------------------------------------------------
    logger.info("Loading data from '%s' …", args.data_dir)
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(
        "Dataset ready | train batches=%d | val batches=%d",
        len(train_loader), len(val_loader),
    )

    # ---- Model -----------------------------------------------------------
    model = TrafficSignNet(num_classes=args.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: TrafficSignNet | total_params=%s | trainable=%s",
        f"{total_params:,}", f"{trainable:,}",
    )

    # ---- Loss, optimiser, scheduler, early stopping ----------------------
    criterion  = nn.CrossEntropyLoss()
    optimiser  = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = ReduceLROnPlateau(
        optimiser, mode="max",      # maximise robust accuracy
        factor=0.5, patience=2, verbose=True,
    )
    early_stop = EarlyStopping(patience=args.patience, min_delta=0.1)

    # ---- PGD attack objects ----------------------------------------------
    # Training PGD (PGD-7): fast enough to run on every batch.
    # torchattacks.PGD expects the model to be in eval mode during generation
    # and handles this automatically; train mode is restored in train_one_epoch.
    pgd_train = torchattacks.PGD(
        model,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
    )

    # Validation PGD (PGD-10): slightly stronger than training attack to
    # provide an honest, unbiased robustness estimate.  Using the same
    # training attack for evaluation can overestimate robustness (the model
    # may have overfit to PGD-7 trajectories).
    pgd_val = torchattacks.PGD(
        model,
        eps=args.eps,
        alpha=args.alpha,
        steps=min(args.steps + 3, 20),   # PGD-10 (or capped at 20)
    )

    # ---- History ---------------------------------------------------------
    history: Dict[str, List[float]] = {
        "adv_train_loss": [],
        "clean_val_acc":  [],
        "robust_val_acc": [],
    }

    best_robust_acc = 0.0
    best_epoch      = 1
    training_start  = time.time()

    logger.info(
        "Starting adversarial training (PGD-%d) | epochs=%d | batch_size=%d | "
        "lr=%.4f | eps=%.4f | alpha=%.4f | patience=%d",
        args.steps, args.epochs, args.batch_size,
        args.lr, args.eps, args.alpha, args.patience,
    )

    # ====================================================================
    # Training loop
    # ====================================================================
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # -- Adversarial training pass ------------------------------------
        adv_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimiser=optimiser,
            pgd=pgd_train,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        # -- Dual validation pass -----------------------------------------
        clean_acc, robust_acc = evaluate(
            model=model,
            loader=val_loader,
            pgd_val=pgd_val,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        # LR scheduler driven by robust accuracy (higher is better)
        scheduler.step(robust_acc)
        current_lr = optimiser.param_groups[0]["lr"]

        history["adv_train_loss"].append(adv_loss)
        history["clean_val_acc"].append(clean_acc)
        history["robust_val_acc"].append(robust_acc)

        epoch_duration = time.time() - epoch_start

        logger.info(
            "Epoch %3d/%d | adv_loss=%.4f | clean_acc=%.2f%% | "
            "robust_acc=%.2f%% | gap=%.2f%% | lr=%.2e | %.1fs",
            epoch, args.epochs,
            adv_loss, clean_acc, robust_acc,
            clean_acc - robust_acc,     # robustness gap
            current_lr, epoch_duration,
        )

        # -- Save best checkpoint on robust accuracy ----------------------
        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            best_epoch      = epoch
            torch.save(model.state_dict(), weights_path)
            logger.info(
                "  ✓ New best robust model saved → %s  "
                "(robust_acc=%.2f%%  clean_acc=%.2f%%)",
                weights_path, robust_acc, clean_acc,
            )

        # -- Early stopping check -----------------------------------------
        if early_stop.step(robust_acc):
            logger.warning(
                "Early stopping triggered at epoch %d. "
                "Robust val acc did not improve for %d consecutive epochs. "
                "Best robust_acc=%.2f%% at epoch %d.",
                epoch, args.patience, best_robust_acc, best_epoch,
            )
            break

    # ====================================================================
    # Training complete – summary & artefacts
    # ====================================================================
    total_duration = time.time() - training_start
    final_gap = (
        history["clean_val_acc"][best_epoch - 1]
        - history["robust_val_acc"][best_epoch - 1]
    )

    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║       AI-SOC Blue Team — Adversarial Training Complete       ║
╠══════════════════════════════════════════════════════════════╣
║  Model          : TrafficSignNet (robust_cnn.pth)            ║
║  Attack (train) : PGD-{args.steps:<3d}  eps={args.eps:.4f}  alpha={args.alpha:.4f}        ║
║  Attack (val)   : PGD-{min(args.steps+3,20):<3d}  (harder eval attack)            ║
║  Duration       : {total_duration/60:.1f} min                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Best epoch          : {best_epoch:<3d}                                 ║
║  Best Clean Val Acc  : {history['clean_val_acc'][best_epoch-1]:>6.2f}%                        ║
║  Best Robust Val Acc : {best_robust_acc:>6.2f}%  ← checkpoint criterion      ║
║  Robustness Gap      : {final_gap:>6.2f}%                        ║
╠══════════════════════════════════════════════════════════════╣
║  Weights → {str(weights_path):<50s}║
║  Plot    → {str(plot_path):<50s}║
╚══════════════════════════════════════════════════════════════╝"""
    logger.info(summary)

    # ---- Visualisation --------------------------------------------------
    save_training_plots(
        adv_train_losses=history["adv_train_loss"],
        clean_val_accs=history["clean_val_acc"],
        robust_val_accs=history["robust_val_acc"],
        best_epoch=best_epoch,
        save_path=plot_path,
    )

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_adv_training(args)

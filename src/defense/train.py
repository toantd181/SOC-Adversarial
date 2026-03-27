"""
src/defense/train.py

Standard Training Pipeline – Baseline CNN (TrafficSignNet)
===========================================================
Trains the Baseline (vulnerable) model on the traffic-sign dataset and saves
the best checkpoint to ``weights/baseline_cnn.pth``.

Usage (Kaggle / Cloud GPU):
    python -m src.defense.train \
        --data_dir  data/processed \
        --epochs    30 \
        --batch_size 64 \
        --lr        0.001 \
        --patience  5

Outputs
-------
* weights/baseline_cnn.pth          – best model weights (lowest val loss)
* logs/training_history.png         – loss & accuracy curves
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")          # headless backend – safe for Kaggle / SSH
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Absolute imports from project root
# ---------------------------------------------------------------------------
from src.data.dataset import get_data_loaders          # returns train/val DataLoaders
from src.models.cnn_classifier import TrafficSignNet  # Victim Model architecture

# ---------------------------------------------------------------------------
# Logging – use stdlib here (no FastAPI dependency in cloud phase)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)-8s] | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ai_soc.train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Standard training of the AI-SOC Baseline CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the processed dataset root (must contain train/ and val/ splits).",
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
        help="Mini-batch size for both train and validation loaders.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for the Adam optimiser.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help=(
            "Early-stopping patience: number of consecutive epochs without "
            "validation-loss improvement before training is halted."
        ),
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=43,
        help="Number of output classes (43 for GTSRB, 10 for CIFAR-10).",
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
        help="Directory where training plots and logs are saved.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    args = parser.parse_args()

    # ---- Sanity checks ---------------------------------------------------
    if args.epochs < 1:
        parser.error("--epochs must be ≥ 1.")
    if args.batch_size < 1:
        parser.error("--batch_size must be ≥ 1.")
    if args.lr <= 0:
        parser.error("--lr must be a positive float.")
    if args.patience < 1:
        parser.error("--patience must be ≥ 1.")

    return args


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor a scalar metric (lower is better) and signal when training
    should stop because the metric has not improved for ``patience`` epochs.

    Attributes
    ----------
    patience : int
        Number of epochs to wait after last improvement before stopping.
    min_delta : float
        Minimum absolute change to qualify as an improvement.
    counter : int
        Number of consecutive epochs without improvement.
    best_score : float | None
        Best metric value observed so far.
    triggered : bool
        True once the patience limit has been exceeded.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_score: float | None = None
        self.triggered: bool = False

    def step(self, metric: float) -> bool:
        """
        Update internal state and return True if training should stop.

        Parameters
        ----------
        metric:
            Current epoch's validation loss (lower is better).

        Returns
        -------
        bool
            True  → stop training now.
            False → continue training.
        """
        if self.best_score is None or metric < self.best_score - self.min_delta:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping counter: %d / %d  (best val_loss=%.6f)",
                self.counter,
                self.patience,
                self.best_score,
            )
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False


# ---------------------------------------------------------------------------
# Training & evaluation passes
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """
    Run one full training epoch.

    Parameters
    ----------
    model       : Network to train.
    loader      : Training DataLoader.
    criterion   : Loss function.
    optimiser   : Gradient-descent optimiser.
    device      : torch.device (cuda / cpu).
    epoch       : Current epoch index (1-based, for display only).
    total_epochs: Maximum epochs (for tqdm description).

    Returns
    -------
    float
        Mean training loss over the epoch.
    """
    model.train()
    running_loss: float = 0.0
    n_batches: int = len(loader)

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:>3}/{total_epochs} [TRAIN]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in progress:
        images: torch.Tensor = images.to(device, non_blocking=True)
        labels: torch.Tensor = labels.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)          # faster than zero_grad()
        logits: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        batch_loss = loss.item()
        running_loss += batch_loss
        progress.set_postfix(loss=f"{batch_loss:.4f}")

    return running_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """
    Run one full validation / evaluation epoch without gradient computation.

    Parameters
    ----------
    model       : Network to evaluate.
    loader      : Validation DataLoader.
    criterion   : Loss function.
    device      : torch.device (cuda / cpu).
    epoch       : Current epoch index (1-based, for display only).
    total_epochs: Maximum epochs (for tqdm description).

    Returns
    -------
    Tuple[float, float]
        (mean_val_loss, val_accuracy_percent)
    """
    model.eval()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0
    n_batches: int = len(loader)

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:>3}/{total_epochs} [VAL]  ",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(logits, labels)

        running_loss += loss.item()

        preds: torch.Tensor = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = running_loss / n_batches
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return mean_loss, accuracy


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    save_path: Path,
) -> None:
    """
    Generate and save a two-subplot training history figure.

    Subplot 1 – Train Loss vs. Validation Loss (line plot, per epoch).
    Subplot 2 – Validation Accuracy % (line plot, per epoch).

    Parameters
    ----------
    train_losses    : Per-epoch mean training losses.
    val_losses      : Per-epoch mean validation losses.
    val_accuracies  : Per-epoch validation accuracy (%).
    save_path       : Absolute / relative path for the output PNG.
    """
    epochs_range = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle("AI-SOC Baseline CNN – Training History", fontsize=14, fontweight="bold")

    # ---- Subplot 1: Loss curves -----------------------------------------
    ax1 = axes[0]
    ax1.plot(epochs_range, train_losses, label="Train Loss",      color="#2196F3", linewidth=2, marker="o", markersize=4)
    ax1.plot(epochs_range, val_losses,   label="Validation Loss", color="#F44336", linewidth=2, marker="s", markersize=4)

    # Mark the epoch with the best (minimum) validation loss
    best_epoch = int(torch.tensor(val_losses).argmin().item()) + 1
    best_val   = min(val_losses)
    ax1.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax1.annotate(
        f"  {best_val:.4f}",
        xy=(best_epoch, best_val),
        color="gray",
        fontsize=9,
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss vs. Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ---- Subplot 2: Validation Accuracy ----------------------------------
    ax2 = axes[1]
    ax2.plot(epochs_range, val_accuracies, label="Val Accuracy (%)", color="#4CAF50", linewidth=2, marker="^", markersize=4)

    best_acc_epoch = int(torch.tensor(val_accuracies).argmax().item()) + 1
    best_acc       = max(val_accuracies)
    ax2.axvline(x=best_acc_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best_acc_epoch})")
    ax2.annotate(
        f"  {best_acc:.2f}%",
        xy=(best_acc_epoch, best_acc),
        color="gray",
        fontsize=9,
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy over Epochs")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                # release memory – critical on long runs
    logger.info("Training history plot saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> Dict[str, List[float]]:
    """
    Orchestrate the full standard training pipeline.

    1. Resolve directories and create them if absent.
    2. Detect hardware device.
    3. Instantiate model, optimiser, loss, scheduler, and early stopper.
    4. Execute the train / validate loop with early stopping.
    5. Save the best weights and training plot.

    Parameters
    ----------
    args : Parsed argparse.Namespace from ``parse_args()``.

    Returns
    -------
    Dict[str, List[float]]
        History dict with keys: ``train_loss``, ``val_loss``, ``val_accuracy``.
    """
    # ---- Directories -----------------------------------------------------
    weights_dir = Path(args.weights_dir)
    logs_dir    = Path(args.logs_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    weights_path = weights_dir / "baseline_cnn.pth"
    plot_path    = logs_dir    / "training_history.png"

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
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(
        "Dataset ready | train batches=%d | val batches=%d",
        len(train_loader),
        len(val_loader),
    )

    # ---- Model -----------------------------------------------------------
    model = TrafficSignNet(num_classes=args.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: TrafficSignNet | total_params=%s | trainable=%s",
        f"{total_params:,}",
        f"{trainable:,}",
    )

    # ---- Loss, optimiser, scheduler --------------------------------------
    criterion  = nn.CrossEntropyLoss()
    optimiser  = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # ReduceLROnPlateau halves LR when val_loss plateaus for 2 epochs,
    # complementing early stopping with a softer first response.
    scheduler  = ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2, verbose=True
    )
    early_stop = EarlyStopping(patience=args.patience, min_delta=1e-4)

    # ---- History buffers -------------------------------------------------
    history: Dict[str, List[float]] = {
        "train_loss":   [],
        "val_loss":     [],
        "val_accuracy": [],
    }

    best_val_loss   = float("inf")
    best_epoch      = 0
    training_start  = time.time()

    logger.info(
        "Starting standard training | epochs=%d | batch_size=%d | lr=%.4f | patience=%d",
        args.epochs, args.batch_size, args.lr, args.patience,
    )

    # ---- Training loop ---------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train pass
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimiser, device, epoch, args.epochs
        )

        # Validation pass
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        # LR scheduler step
        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        epoch_duration = time.time() - epoch_start

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% | lr=%.2e | %.1fs",
            epoch, args.epochs,
            train_loss, val_loss, val_acc,
            current_lr, epoch_duration,
        )

        # ---- Save best checkpoint ----------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), weights_path)
            logger.info(
                "  ✓ New best model saved → %s  (val_loss=%.6f)",
                weights_path, best_val_loss,
            )

        # ---- Early stopping check ----------------------------------------
        if early_stop.step(val_loss):
            logger.warning(
                "Early stopping triggered at epoch %d. "
                "Val loss did not improve for %d consecutive epochs. "
                "Best val_loss=%.6f at epoch %d.",
                epoch, args.patience, best_val_loss, best_epoch,
            )
            break

    # ---- Training summary ------------------------------------------------
    total_duration = time.time() - training_start
    logger.info("=" * 60)
    logger.info("Training complete in %.1f s (%.1f min)", total_duration, total_duration / 60)
    logger.info("Best val_loss : %.6f  (epoch %d)", best_val_loss, best_epoch)
    logger.info("Best val_acc  : %.2f%%", max(history["val_accuracy"]))
    logger.info("Weights saved : %s", weights_path)
    logger.info("=" * 60)

    # ---- Visualisation ---------------------------------------------------
    save_training_plots(
        train_losses=history["train_loss"],
        val_losses=history["val_loss"],
        val_accuracies=history["val_accuracy"],
        save_path=plot_path,
    )

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_training(args)
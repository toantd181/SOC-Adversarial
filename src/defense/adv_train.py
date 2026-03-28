"""
src/defense/adv_train.py

Adversarial Training Pipeline – Robust CNN (TrafficSignNet)
===========================================================

The AI Vaccine Analogy
-----------------------
Standard training teaches a model to classify "healthy" inputs — images the
world naturally produces.  Adversarial training is the immunisation shot: we
deliberately inject worst-case perturbations (PGD adversarial examples) into
every training batch, forcing the model's decision boundaries to harden
against them.

Just as a vaccine exposes the immune system to an attenuated pathogen so it
learns to recognise and defeat the real virus, adversarial training exposes
the model to the strongest attacks it will face during inference.  At
deployment time, the adversary can no longer trivially shift predictions
because the model has already learned to resist those exact perturbation
trajectories.

Warm-Starting from the Baseline
---------------------------------
Training a robust model from random initialisation is slow and unstable.
By loading the pre-trained baseline weights (``--baseline_weights``) first,
we inherit all feature representations already learned from clean data and
fine-tune only the decision boundaries — analogous to boosting an existing
immune response rather than building one from scratch.  The lower default
learning rate (1e-4 vs 1e-3 in standard training) reflects this fine-tuning
regime: large gradient steps would destabilise the inherited representations.

PGD (Projected Gradient Descent) – Madry et al., 2018
-------------------------------------------------------
PGD is the strong iterative adversary used for both Red Team evaluation and
Blue Team defence.  At each of ``steps`` iterations it takes a step of size
``alpha`` in the direction that maximises the cross-entropy loss, then
projects the cumulative perturbation back onto the L∞ ball of radius ``eps``
around the original input.  PGD-7 during training is the community standard;
a harder PGD-10 is used for validation to provide an unbiased robustness
estimate that does not overfit to the training attack's exact trajectory.

Training strategy: pure adversarial batches
--------------------------------------------
This script trains exclusively on PGD adversarial examples.  The warm-start
from baseline weights supplies clean-data feature representations, so no
additional clean gradient signal is needed during fine-tuning.  Pure
adversarial batches push robust accuracy higher at the cost of a small clean
accuracy drop — the correct trade-off when the primary deployment concern is
adversarial robustness.

Checkpoint & early-stopping criterion: robust_val_loss
-------------------------------------------------------
Robust validation loss is a smoother, more informative signal than robust
accuracy (a step function): it captures both prediction correctness and
model confidence under attack, so small robustness improvements register
even when accuracy stays flat.  Both the best checkpoint and the early
stopping counter are driven by this metric.

Usage (Kaggle / Cloud GPU):
    python -m src.defense.adv_train \\
        --data_dir          data/processed \\
        --baseline_weights  weights/baseline_cnn.pth \\
        --epochs            30 \\
        --batch_size        64 \\
        --lr                1e-4 \\
        --patience          6 \\
        --eps               0.03137 \\
        --alpha             0.00784 \\
        --steps             7
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
from src.data.dataset import get_data_loaders
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
SOC_BLUE    = "#2196F3"   # adversarial train loss line
SOC_GREEN   = "#4CAF50"   # clean validation accuracy line
SOC_ORANGE  = "#FF9800"   # robust validation accuracy line


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for adversarial training.

    PGD hyperparameter guidance
    ----------------------------
    eps   – L∞ budget calibrated to the Red Team evaluation (8/255) so the
            defence is trained against exactly the threat it will face.
    alpha – Per-step size; Madry et al. recommend alpha ≈ eps / 4, giving
            the PGD optimiser enough range to explore the ε-ball without
            overshooting in a small number of steps.
    steps – PGD-7 balances attack strength against per-batch compute cost;
            training with a weaker attack (PGD-3) produces brittle models
            that fail against stronger evaluation adversaries.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial Training (PGD-AT) of the AI-SOC Robust CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the processed dataset (train/ and val/ splits).",
    )
    parser.add_argument(
        "--baseline_weights",
        type=str,
        default="weights/baseline_cnn.pth",
        help=(
            "Path to baseline model weights used for warm-starting.  "
            "Inheriting clean-data representations accelerates convergence "
            "and yields higher robust accuracy than training from scratch."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of adversarial training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate (fine-tuning regime; lower than standard training).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early-stopping patience: epochs without robust_val_loss improvement.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=8 / 255,
        help="PGD L∞ perturbation budget ε.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2 / 255,
        help="PGD per-step size α.  Recommended: eps / 4.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=7,
        help="PGD iteration count per batch (PGD-7 is the standard training attack).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=43,
        help="Number of output classes (43 for GTSRB).",
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

    # ---- Sanity checks ---------------------------------------------------
    if args.epochs < 1:
        parser.error("--epochs must be ≥ 1.")
    if args.batch_size < 1:
        parser.error("--batch_size must be ≥ 1.")
    if args.lr <= 0.0:
        parser.error("--lr must be a positive float.")
    if args.patience < 1:
        parser.error("--patience must be ≥ 1.")
    if not (0.0 < args.eps <= 1.0):
        parser.error("--eps must be in (0, 1].")
    if not (0.0 < args.alpha <= args.eps):
        parser.error("--alpha must be in (0, eps].")
    if args.steps < 1:
        parser.error("--steps must be ≥ 1.")

    return args


# ---------------------------------------------------------------------------
# Model loading – warm-start from baseline
# ---------------------------------------------------------------------------

def build_model(
    num_classes:      int,
    baseline_weights: str,
    device:           torch.device,
) -> nn.Module:
    """
    Instantiate TrafficSignNet and warm-start from baseline weights.

    Parameters
    ----------
    num_classes       : Output head dimension.
    baseline_weights  : Path to the state-dict saved by ``train.py``.
    device            : Target device (cuda / cpu).

    Returns
    -------
    nn.Module
        Model with baseline weights loaded, on ``device``, in train mode.

    Raises
    ------
    FileNotFoundError
        If ``baseline_weights`` does not exist on disk.
    """
    path = Path(baseline_weights)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline weights not found: '{path.resolve()}'. "
            "Run src/defense/train.py first to produce baseline_cnn.pth."
        )

    model = TrafficSignNet(num_classes=num_classes)
    # map_location prevents GPU→CPU device-mismatch errors when weights
    # were saved on a different hardware configuration
    state_dict = torch.load(str(path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()

    logger.info(
        "Warm-start: baseline weights loaded from '%s' onto device '%s'.",
        path, device,
    )
    return model


# ---------------------------------------------------------------------------
# Early stopping  (monitors robust_val_loss – lower is better)
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Halt training when ``robust_val_loss`` fails to decrease by at least
    ``min_delta`` for ``patience`` consecutive epochs.

    Loss is preferred over accuracy as the stopping criterion because it is
    a smooth, continuous signal that reflects both prediction correctness and
    model confidence under attack.  Small robustness improvements are
    captured even when the discrete accuracy metric remains flat.

    Parameters
    ----------
    patience  : Consecutive epochs without improvement before stopping.
    min_delta : Minimum absolute loss decrease that counts as improvement.
    """

    def __init__(self, patience: int = 6, min_delta: float = 1e-4) -> None:
        self.patience:  int          = patience
        self.min_delta: float        = min_delta
        self.counter:   int          = 0
        self.best_loss: float | None = None
        self.triggered: bool         = False

    def step(self, loss: float) -> bool:
        """
        Update internal state with the current epoch's robust_val_loss.

        Returns
        -------
        bool
            ``True``  → stop training.
            ``False`` → continue.
        """
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter   = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping: %d / %d  (best robust_val_loss=%.6f)",
                self.counter, self.patience, self.best_loss,
            )
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False


# ---------------------------------------------------------------------------
# Training pass  (pure adversarial — PGD on every batch)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    criterion:    nn.Module,
    optimiser:    torch.optim.Optimizer,
    pgd_train:    torchattacks.PGD,
    device:       torch.device,
    epoch:        int,
    total_epochs: int,
) -> float:
    """
    Execute one full adversarial training epoch.

    For every mini-batch the function:
      1. Generates PGD adversarial examples **on the fly** for the entire
         batch (no pre-computation or caching on disk).
      2. Restores ``model.train()`` immediately after the attack call.
         ``torchattacks`` internally calls ``model.eval()`` before running
         the attack to disable dropout stochasticity and force BatchNorm into
         inference mode — both necessary for a stable loss gradient.  Without
         this restore call, the subsequent training forward pass would silently
         run in eval mode, breaking BatchNorm statistics and disabling dropout.
      3. Runs a standard forward / backward pass using **only** the adversarial
         examples — no clean images are seen by the optimiser in this phase.

    Parameters
    ----------
    model        : Network being trained.
    loader       : Training DataLoader.
    criterion    : Cross-entropy loss.
    optimiser    : Adam optimiser.
    pgd_train    : ``torchattacks.PGD`` configured for the training budget.
    device       : Target device.
    epoch        : Current epoch index (1-based), for tqdm display.
    total_epochs : Maximum epoch count, for tqdm display.

    Returns
    -------
    float
        Mean adversarial training loss over all batches in this epoch.
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

        # ---- On-the-fly PGD adversarial example generation ---------------
        adv_images: torch.Tensor = pgd_train(images, labels)
        model.train()   # restore after torchattacks sets model.eval()

        # ---- Forward / backward on adversarial examples only -------------
        optimiser.zero_grad(set_to_none=True)
        logits: torch.Tensor = model(adv_images)
        loss:   torch.Tensor = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        batch_loss    = loss.item()
        running_loss += batch_loss
        progress.set_postfix(adv_loss=f"{batch_loss:.4f}")

    return running_loss / n_batches


# ---------------------------------------------------------------------------
# Dual validation pass
# ---------------------------------------------------------------------------

def evaluate(
    model:        nn.Module,
    loader:       DataLoader,
    criterion:    nn.Module,
    pgd_val:      torchattacks.PGD,
    device:       torch.device,
    epoch:        int,
    total_epochs: int,
) -> Tuple[float, float, float]:
    """
    Evaluate on both the clean validation set and a PGD-attacked version.

    Two separate passes are required:
      * **Clean pass** – full ``torch.no_grad()`` scope; no attack computation.
      * **Robust pass** – autograd must be live for the PGD optimiser;
        ``torch.no_grad()`` is applied only around the post-attack forward
        pass used for accuracy and loss measurement.

    Parameters
    ----------
    model        : Network to evaluate (set to eval mode internally).
    loader       : Validation DataLoader.
    criterion    : Cross-entropy loss (used to compute robust_val_loss).
    pgd_val      : ``torchattacks.PGD`` for validation (typically PGD-10,
                   harder than the training PGD-7 for an unbiased estimate).
    device       : Target device.
    epoch        : Current epoch index (1-based), for tqdm display.
    total_epochs : Maximum epoch count, for tqdm display.

    Returns
    -------
    Tuple[float, float, float]
        ``(clean_val_acc, robust_val_acc, robust_val_loss)``
        Accuracies are percentages; loss is mean cross-entropy.
    """
    model.eval()

    # ---- Pass 1: clean accuracy ------------------------------------------
    clean_correct: int = 0
    clean_total:   int = 0

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

    # ---- Pass 2: robust accuracy and robust loss (PGD needs autograd) ----
    robust_correct:    int   = 0
    robust_total:      int   = 0
    robust_loss_accum: float = 0.0
    n_batches:         int   = len(loader)

    for images, labels in tqdm(
        loader,
        desc=f"Epoch {epoch:>3}/{total_epochs} [VAL-ROBUST]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # PGD attack – autograd must be active here
        adv_images: torch.Tensor = pgd_val(images, labels)

        # Measurement only – no weight updates
        with torch.no_grad():
            logits = model(adv_images)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(dim=1)

        robust_correct    += int((preds == labels).sum().item())
        robust_total      += labels.size(0)
        robust_loss_accum += loss.item()

    robust_acc  = 100.0 * robust_correct / robust_total if robust_total > 0 else 0.0
    robust_loss = robust_loss_accum / n_batches

    return clean_acc, robust_acc, robust_loss


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_training_plots(
    adv_train_losses: List[float],
    clean_val_accs:   List[float],
    robust_val_accs:  List[float],
    best_epoch:       int,
    save_path:        Path,
) -> None:
    """
    Render and save the adversarial training history figure.

    Layout
    ------
    Dual y-axis plot on a shared epoch x-axis:
      * Left axis  (blue)   – Adversarial train loss; lower = better.
      * Right axis (green)  – Clean validation accuracy %; higher = better.
      * Right axis (orange) – Robust validation accuracy %; higher = better.

    The shaded fill between the green and orange curves is the **robustness
    gap** — how much clean accuracy the model trades for adversarial
    resilience.  Watching this gap narrow over epochs is the visual proof
    that adversarial training is taking effect.

    A vertical dashed line marks the best checkpoint epoch (lowest
    robust_val_loss).  Annotations show both accuracy values at that epoch.

    Parameters
    ----------
    adv_train_losses : Per-epoch mean adversarial training loss.
    clean_val_accs   : Per-epoch clean validation accuracy (%).
    robust_val_accs  : Per-epoch robust validation accuracy (%).
    best_epoch       : 1-based epoch index of the saved checkpoint.
    save_path        : Destination PNG path.
    """
    epochs_range = range(1, len(adv_train_losses) + 1)

    fig, ax_loss = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(SOC_BG)
    ax_loss.set_facecolor(SOC_SURFACE)

    ax_loss.yaxis.grid(True, color=SOC_GRID, linewidth=0.7, linestyle="--", zorder=0)
    ax_loss.set_axisbelow(True)

    # ---- Left axis: adversarial training loss ----------------------------
    loss_line, = ax_loss.plot(
        epochs_range, adv_train_losses,
        label="Adv. Train Loss",
        color=SOC_BLUE, linewidth=2,
        marker="o", markersize=4, zorder=3,
    )
    ax_loss.set_xlabel("Epoch", fontsize=11, color=SOC_TEXT, labelpad=8)
    ax_loss.set_ylabel("Adversarial Training Loss", fontsize=11,
                        color=SOC_BLUE, labelpad=8)
    ax_loss.tick_params(axis="y", colors=SOC_BLUE,  labelsize=9)
    ax_loss.tick_params(axis="x", colors=SOC_TEXT, labelsize=9)

    # ---- Right axis: clean and robust accuracy ---------------------------
    ax_acc = ax_loss.twinx()
    ax_acc.set_facecolor("none")

    clean_line, = ax_acc.plot(
        epochs_range, clean_val_accs,
        label="Clean Val Acc (%)",
        color=SOC_GREEN, linewidth=2,
        marker="s", markersize=4, zorder=3,
    )
    robust_line, = ax_acc.plot(
        epochs_range, robust_val_accs,
        label="Robust Val Acc (%)",
        color=SOC_ORANGE, linewidth=2,
        marker="^", markersize=4, zorder=3,
    )

    # Robustness gap shading
    ax_acc.fill_between(
        epochs_range,
        clean_val_accs,
        robust_val_accs,
        alpha=0.12,
        color=SOC_ORANGE,
        zorder=2,
    )

    ax_acc.set_ylabel("Validation Accuracy (%)", fontsize=11,
                       color=SOC_TEXT, labelpad=8)
    ax_acc.tick_params(axis="y", colors=SOC_SUBTEXT, labelsize=9)
    ax_acc.set_ylim(0, 110)

    # ---- Best checkpoint vertical marker ---------------------------------
    best_robust = robust_val_accs[best_epoch - 1]
    best_clean  = clean_val_accs[best_epoch - 1]

    ax_acc.axvline(
        x=best_epoch,
        color=SOC_ORANGE, linestyle="--",
        linewidth=1.2, alpha=0.75, zorder=4,
    )
    ax_acc.annotate(
        f"  Best epoch {best_epoch}\n"
        f"  Robust: {best_robust:.2f}%\n"
        f"  Clean:  {best_clean:.2f}%",
        xy=(best_epoch, best_robust),
        color=SOC_ORANGE, fontsize=8.5,
    )

    # ---- Spines ----------------------------------------------------------
    for ax in (ax_loss, ax_acc):
        for spine in ax.spines.values():
            spine.set_color(SOC_GRID)
            spine.set_linewidth(0.8)

    # ---- Combined legend -------------------------------------------------
    handles = [loss_line, clean_line, robust_line]
    ax_loss.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper right",
        fontsize=9,
        framealpha=0.25,
        facecolor=SOC_BG,
        edgecolor=SOC_GRID,
        labelcolor=SOC_TEXT,
    )

    # ---- Titles ----------------------------------------------------------
    fig.suptitle(
        "AI-SOC Blue Team — Adversarial Training History  (PGD-AT, warm-start)",
        fontsize=13, fontweight="bold", color=SOC_TEXT, y=0.98,
    )
    ax_loss.set_title(
        "Shaded area = robustness gap (clean acc − robust acc).  "
        "Dashed line = best checkpoint (lowest robust_val_loss).",
        fontsize=9, color=SOC_SUBTEXT, pad=6,
    )

    # ---- Save (no plt.show – headless environment) -----------------------
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        str(save_path), dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    logger.info("Adversarial training history plot saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_adv_training(args: argparse.Namespace) -> Dict[str, List[float]]:
    """
    Orchestrate the full adversarial training pipeline.

    Steps
    -----
    1. Resolve and create output directories.
    2. Detect hardware device.
    3. Load data; warm-start model from baseline weights.
    4. Build optimiser, scheduler, PGD attack objects, and early stopper.
    5. Execute the adversarial train / dual-validate loop.
    6. Save the best robust checkpoint and the training history plot.

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
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(
        "Dataset ready | train batches=%d | val batches=%d",
        len(train_loader), len(val_loader),
    )

    # ---- Model (warm-start) ----------------------------------------------
    model = build_model(
        num_classes=args.num_classes,
        baseline_weights=args.baseline_weights,
        device=device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "TrafficSignNet | total_params=%s | trainable=%s",
        f"{total_params:,}", f"{trainable:,}",
    )

    # ---- Loss, optimiser -------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ReduceLROnPlateau monitoring robust_val_loss (lower is better → mode='min').
    # verbose=True is deprecated in recent PyTorch; LR changes are logged
    # manually by comparing param_groups before and after scheduler.step().
    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=0.5,
        patience=2,
    )

    early_stop = EarlyStopping(patience=args.patience, min_delta=1e-4)

    # ---- PGD attack objects ----------------------------------------------
    # Training attack: PGD-7 – fast enough for per-batch on-the-fly generation.
    pgd_train = torchattacks.PGD(
        model,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
    )

    # Validation attack: PGD-10 (3 extra steps) – harder than the training
    # attack to prevent the robustness estimate from being overfit to PGD-7.
    val_steps = min(args.steps + 3, 20)
    pgd_val   = torchattacks.PGD(
        model,
        eps=args.eps,
        alpha=args.alpha,
        steps=val_steps,
    )

    logger.info(
        "PGD-train: steps=%d | PGD-val: steps=%d | eps=%.4f | alpha=%.4f",
        args.steps, val_steps, args.eps, args.alpha,
    )

    # ---- History buffers -------------------------------------------------
    history: Dict[str, List[float]] = {
        "adv_train_loss": [],
        "clean_val_acc":  [],
        "robust_val_acc": [],
    }

    best_robust_loss: float = float("inf")
    best_epoch:       int   = 1
    training_start:   float = time.time()

    logger.info(
        "Starting PGD-%d adversarial training | epochs=%d | batch_size=%d "
        "| lr=%.1e | eps=%.4f | alpha=%.4f | patience=%d",
        args.steps, args.epochs, args.batch_size,
        args.lr, args.eps, args.alpha, args.patience,
    )

    # ====================================================================
    # Main training loop
    # ====================================================================
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # -- Adversarial training pass ------------------------------------
        adv_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimiser=optimiser,
            pgd_train=pgd_train,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        # -- Dual validation pass -----------------------------------------
        clean_acc, robust_acc, robust_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            pgd_val=pgd_val,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        # -- LR scheduler step on robust_val_loss (manual verbose logging) -
        prev_lr = optimiser.param_groups[0]["lr"]
        scheduler.step(robust_loss)
        current_lr = optimiser.param_groups[0]["lr"]
        if current_lr < prev_lr:
            logger.info(
                "ReduceLROnPlateau: LR reduced %.2e → %.2e",
                prev_lr, current_lr,
            )

        # -- Record history ------------------------------------------------
        history["adv_train_loss"].append(adv_loss)
        history["clean_val_acc"].append(clean_acc)
        history["robust_val_acc"].append(robust_acc)

        epoch_duration = time.time() - epoch_start

        logger.info(
            "Epoch %3d/%d | adv_loss=%.4f | robust_loss=%.4f | "
            "clean_acc=%.2f%% | robust_acc=%.2f%% | gap=%.2f%% | lr=%.2e | %.1fs",
            epoch, args.epochs,
            adv_loss, robust_loss,
            clean_acc, robust_acc,
            clean_acc - robust_acc,     # robustness gap
            current_lr, epoch_duration,
        )

        # -- Save best checkpoint based on robust_val_loss -----------------
        if robust_loss < best_robust_loss:
            best_robust_loss = robust_loss
            best_epoch       = epoch
            torch.save(model.state_dict(), weights_path)
            logger.info(
                "  ✓ Best robust model saved → %s  "
                "(robust_loss=%.6f | robust_acc=%.2f%% | clean_acc=%.2f%%)",
                weights_path, robust_loss, robust_acc, clean_acc,
            )

        # -- Early stopping check ------------------------------------------
        if early_stop.step(robust_loss):
            logger.warning(
                "Early stopping triggered at epoch %d. "
                "robust_val_loss did not improve for %d consecutive epochs. "
                "Best robust_val_loss=%.6f at epoch %d.",
                epoch, args.patience, best_robust_loss, best_epoch,
            )
            break

    # ====================================================================
    # Post-training summary & artefacts
    # ====================================================================
    total_duration  = time.time() - training_start
    best_robust_acc = history["robust_val_acc"][best_epoch - 1]
    best_clean_acc  = history["clean_val_acc"][best_epoch - 1]
    robustness_gap  = best_clean_acc - best_robust_acc

    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║       AI-SOC Blue Team — Adversarial Training Complete       ║
╠══════════════════════════════════════════════════════════════╣
║  Model            : TrafficSignNet (robust_cnn.pth)          ║
║  Warm-start from  : {args.baseline_weights:<42s}║
║  Attack (train)   : PGD-{args.steps:<3d}  eps={args.eps:.4f}  alpha={args.alpha:.4f}      ║
║  Attack (val)     : PGD-{val_steps:<3d}  (harder evaluation attack)        ║
║  Duration         : {total_duration / 60:>5.1f} min                              ║
╠══════════════════════════════════════════════════════════════╣
║  Best epoch            : {best_epoch:<3d}                               ║
║  Best Robust Val Loss  : {best_robust_loss:>8.6f}  ← checkpoint criterion    ║
║  Best Clean Val Acc    : {best_clean_acc:>6.2f}%                          ║
║  Best Robust Val Acc   : {best_robust_acc:>6.2f}%                          ║
║  Robustness Gap        : {robustness_gap:>6.2f}%                          ║
╠══════════════════════════════════════════════════════════════╣
║  Weights → {str(weights_path):<50s}║
║  Plot    → {str(plot_path):<50s}║
╚══════════════════════════════════════════════════════════════╝"""
    logger.info(summary)

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

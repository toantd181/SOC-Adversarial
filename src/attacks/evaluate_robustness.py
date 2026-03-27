"""
src/attacks/evaluate_robustness.py

Red Team Robustness Evaluator – AI-SOC Adversarial Defense System
==================================================================
Quantifies the vulnerability of the Baseline CNN (TrafficSignNet) against
two fundamentally different classes of adversarial attack:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  FGSM  (Fast Gradient Sign Method)  – Goodfellow et al., 2014      │
  │  ─────────────────────────────────────────────────────────────────  │
  │  Single-step, gradient-based attack.  Computes the sign of the      │
  │  loss gradient w.r.t. the input and perturbs every pixel by ±ε in  │
  │  one shot.  Extremely fast (O(1) backward pass) but produces        │
  │  relatively coarse perturbations that are detectable by statistical  │
  │  anomaly detectors.  Ideal for cheap, high-throughput red-teaming.  │
  │                                                                     │
  │  C&W   (Carlini & Wagner L2 Attack)  – Carlini & Wagner, 2017      │
  │  ─────────────────────────────────────────────────────────────────  │
  │  Iterative optimisation-based attack.  Jointly minimises the L2     │
  │  perturbation norm and a classification objective via Adam.  Needs  │
  │  many forward/backward passes (``steps`` iterations per sample)     │
  │  but produces near-imperceptible, highly transferable adversarial   │
  │  examples that bypass many defences.  The gold standard for         │
  │  evaluating true model robustness.                                  │
  └─────────────────────────────────────────────────────────────────────┘

Pipeline
--------
1. Clean Accuracy      – baseline performance on unperturbed test set.
2. FGSM Robust Accuracy – performance after fast single-step attack.
3. C&W  Robust Accuracy – performance after strong optimisation attack.
4. Side-by-side visualisation grid saved to logs/adversarial_samples.png.

Usage (Kaggle / Cloud GPU):
    python -m src.attacks.evaluate_robustness \
        --data_dir   data/processed \
        --weights_path weights/baseline_cnn.pth \
        --batch_size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # headless – no display server on Kaggle / SSH
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torchattacks
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.api.logger import get_logger
from src.data.dataset import get_test_loader
from src.models.cnn_classifier import TrafficSignNet

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# GTSRB class names (43 classes).  Indices match the dataset label encoding.
# ---------------------------------------------------------------------------

GTSRB_LABELS: Dict[int, str] = {
    0:  "Speed 20",   1:  "Speed 30",   2:  "Speed 50",
    3:  "Speed 60",   4:  "Speed 70",   5:  "Speed 80",
    6:  "End 80",     7:  "Speed 100",  8:  "Speed 120",
    9:  "No passing", 10: "No pass >3.5t",
    11: "Right-of-way", 12: "Priority road", 13: "Yield",
    14: "Stop",       15: "No vehicles", 16: "No trucks",
    17: "No entry",   18: "General caution",
    19: "Left curve", 20: "Right curve", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road",
    24: "Narrows right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians",
    28: "Children",   29: "Bicycles",   30: "Ice/snow",
    31: "Wild animals", 32: "End restrictions",
    33: "Turn right", 34: "Turn left",  35: "Ahead only",
    36: "Straight/right", 37: "Straight/left",
    38: "Keep right", 39: "Keep left",  40: "Roundabout",
    41: "End no passing", 42: "End no pass >3.5t",
}


def label_name(idx: int, num_classes: int = 43) -> str:
    """Return the human-readable GTSRB class name, or a numeric fallback."""
    if num_classes != 43:
        return f"Class {idx}"
    return GTSRB_LABELS.get(idx, f"Class {idx}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Red Team robustness evaluation of the AI-SOC Baseline CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="weights/baseline_cnn.pth",
        help="Path to the saved TrafficSignNet weights (.pth).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the processed test dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the test DataLoader (keep ≤32 – C&W is memory-intensive).",
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
        "--fgsm_eps",
        type=float,
        default=8 / 255,
        help="FGSM perturbation budget ε (L∞ norm bound).",
    )
    parser.add_argument(
        "--cw_c",
        type=float,
        default=1.0,
        help="C&W trade-off constant c (higher → stronger attack, larger perturbation).",
    )
    parser.add_argument(
        "--cw_steps",
        type=int,
        default=50,
        help="Number of Adam optimisation steps for the C&W attack.",
    )
    parser.add_argument(
        "--vis_samples",
        type=int,
        default=8,
        help="Number of sample images shown in the visualisation grid (≤ batch_size).",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Directory where visualisation artefacts are saved.",
    )
    args = parser.parse_args()

    args.vis_samples = min(args.vis_samples, args.batch_size)
    return args


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Instantiate TrafficSignNet and load pre-trained weights safely.

    Parameters
    ----------
    weights_path : Path to the ``.pth`` state-dict file.
    num_classes  : Number of classification heads.
    device       : Target device (cuda / cpu).

    Returns
    -------
    nn.Module
        Model in eval mode, moved to ``device``.

    Raises
    ------
    FileNotFoundError
        If the weights file does not exist at ``weights_path``.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Weights file not found: '{path.resolve()}'. "
            "Run src/defense/train.py first."
        )

    model = TrafficSignNet(num_classes=num_classes)
    # Safe weight loading: map_location prevents GPU→CPU mismatch errors
    state_dict = torch.load(str(path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Model loaded from '%s' on device '%s'.", path, device)
    return model


# ---------------------------------------------------------------------------
# Accuracy evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[torch.Tensor], List[torch.Tensor]]:
    """
    Evaluate clean (unperturbed) accuracy on the test loader.

    Returns
    -------
    Tuple[float, List[Tensor], List[Tensor]]
        (accuracy_percent, first_batch_images, first_batch_labels)
        The first-batch tensors are retained for the visualisation grid.
    """
    correct = 0
    total   = 0
    snapshot_images: Optional[torch.Tensor] = None
    snapshot_labels: Optional[torch.Tensor] = None

    for batch_idx, (images, labels) in enumerate(
        tqdm(loader, desc="[CLEAN] Evaluating", unit="batch", dynamic_ncols=True)
    ):
        images = images.to(device)
        labels = labels.to(device)

        logits  = model(images)
        preds   = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total   += labels.size(0)

        # Snapshot the first batch for visualisation
        if batch_idx == 0:
            snapshot_images = images.cpu()
            snapshot_labels = labels.cpu()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, snapshot_images, snapshot_labels  # type: ignore[return-value]


def evaluate_adversarial(
    model: nn.Module,
    loader: DataLoader,
    attack: torchattacks.Attack,
    device: torch.device,
    phase_name: str,
    snapshot_images: torch.Tensor,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Generate adversarial examples with ``attack`` and measure robust accuracy.

    ``torchattacks`` calls ``model.forward`` internally and requires the model
    to be in eval mode with gradients available (no ``torch.no_grad()`` wrapper
    around the attack call itself – only around the clean forward pass after).

    Parameters
    ----------
    model           : The victim model (eval mode).
    loader          : Test DataLoader.
    attack          : Instantiated ``torchattacks.Attack`` object.
    device          : Target device.
    phase_name      : Display label for the tqdm bar.
    snapshot_images : The same first-batch images used in the clean snapshot,
                      so we can show clean vs adversarial pairs for the exact
                      same inputs in the visualisation grid.

    Returns
    -------
    Tuple[float, Tensor, Tensor]
        (robust_accuracy_percent,
         adversarial_snapshot_images,
         predicted_labels_for_snapshot)
    """
    correct = 0
    total   = 0
    adv_snapshot: Optional[torch.Tensor]  = None
    adv_preds:    Optional[torch.Tensor]  = None

    for batch_idx, (images, labels) in enumerate(
        tqdm(loader, desc=f"[{phase_name}] Attacking", unit="batch", dynamic_ncols=True)
    ):
        images = images.to(device)
        labels = labels.to(device)

        # Generate adversarial examples – torchattacks handles grad context
        adv_images: torch.Tensor = attack(images, labels)

        # Evaluate model on adversarial examples (no grad needed here)
        with torch.no_grad():
            logits = model(adv_images)
            preds  = logits.argmax(dim=1)

        correct += int((preds == labels).sum().item())
        total   += labels.size(0)

        # Snapshot: perturb the SAME first-batch inputs used for clean snapshot
        if batch_idx == 0:
            adv_first = attack(snapshot_images.to(device), labels[: snapshot_images.size(0)])
            with torch.no_grad():
                snap_preds = model(adv_first).argmax(dim=1)
            adv_snapshot = adv_first.cpu()
            adv_preds    = snap_preds.cpu()

    robust_accuracy = 100.0 * correct / total if total > 0 else 0.0
    return robust_accuracy, adv_snapshot, adv_preds  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised CHW image tensor to an HWC uint8 numpy array
    suitable for ``imshow``.

    Handles both standard [0, 1] and arbitrary normalised ranges by clipping
    after converting to [0, 1] via min-max scaling per image.
    """
    img = tensor.detach().float()
    # Min-max normalise to [0, 1] so display is not affected by normalisation
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img.clamp(0.0, 1.0)

    img_np = img.permute(1, 2, 0).numpy()   # CHW → HWC
    img_np = np.clip(img_np, 0.0, 1.0)
    return (img_np * 255).astype(np.uint8)


def save_adversarial_grid(
    clean_images:    torch.Tensor,
    clean_labels:    torch.Tensor,
    fgsm_images:     torch.Tensor,
    fgsm_preds:      torch.Tensor,
    cw_images:       torch.Tensor,
    cw_preds:        torch.Tensor,
    clean_preds:     torch.Tensor,
    n_samples:       int,
    num_classes:     int,
    save_path:       Path,
) -> None:
    """
    Render a three-row comparison grid and save to ``save_path``.

    Row 1 – Clean images  with the model's correct prediction.
    Row 2 – FGSM examples with the model's (mis)prediction.
    Row 3 – C&W  examples with the model's (mis)prediction.

    A red title indicates a misclassification; green indicates the model
    survived the attack for that sample.

    Parameters
    ----------
    clean_images  : (N, C, H, W) clean image batch.
    clean_labels  : (N,) ground-truth class indices.
    fgsm_images   : (N, C, H, W) FGSM adversarial images.
    fgsm_preds    : (N,) model predictions on FGSM images.
    cw_images     : (N, C, H, W) C&W adversarial images.
    cw_preds      : (N,) model predictions on C&W images.
    clean_preds   : (N,) model predictions on clean images.
    n_samples     : Number of columns (image samples) to display.
    num_classes   : Total classes (for label lookup).
    save_path     : Output file path.
    """
    n = min(n_samples, clean_images.size(0))
    rows = 3   # Clean / FGSM / C&W
    row_labels = ["Clean", "FGSM\n(ε=8/255)", "C&W\n(L2, 50 steps)"]

    fig = plt.figure(figsize=(n * 2.4, rows * 3.0))
    fig.patch.set_facecolor("#0f0f14")           # dark SOC-terminal aesthetic

    gs = gridspec.GridSpec(
        rows, n,
        figure=fig,
        hspace=0.55,
        wspace=0.08,
        left=0.07, right=0.97,
        top=0.88,  bottom=0.04,
    )

    # Row-label annotation on the left margin
    for row_idx, row_label in enumerate(row_labels):
        fig.text(
            0.01,
            1.0 - (row_idx + 0.5) / rows,
            row_label,
            va="center", ha="left",
            fontsize=9, fontweight="bold",
            color="#e0e0e0",
            rotation=90,
        )

    rows_data = [
        (clean_images,  clean_preds,  clean_labels,  True),   # row 0: clean
        (fgsm_images,   fgsm_preds,   clean_labels,  False),  # row 1: FGSM
        (cw_images,     cw_preds,     clean_labels,  False),  # row 2: C&W
    ]

    for row_idx, (images, preds, gt_labels, is_clean_row) in enumerate(rows_data):
        for col_idx in range(n):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor("#0f0f14")

            # Image
            img_np = _tensor_to_display(images[col_idx])
            ax.imshow(img_np, interpolation="nearest")
            ax.axis("off")

            pred_idx = int(preds[col_idx].item())
            gt_idx   = int(gt_labels[col_idx].item())
            pred_str = label_name(pred_idx, num_classes)
            gt_str   = label_name(gt_idx,   num_classes)
            correct  = pred_idx == gt_idx

            if is_clean_row:
                # Clean row: show ground-truth + prediction
                title       = f"GT: {gt_str}\n→ {pred_str}"
                title_color = "#69f0ae" if correct else "#ff5252"
            else:
                # Attack rows: show adversarial prediction vs ground truth
                title       = f"→ {pred_str}"
                title_color = "#69f0ae" if correct else "#ff5252"

            ax.set_title(
                title,
                fontsize=6.5,
                color=title_color,
                pad=3,
                fontweight="bold",
            )

    # Main title
    fig.suptitle(
        "AI-SOC Red Team — Adversarial Attack Visualisation\n"
        "Green title = model survived  |  Red title = model fooled",
        fontsize=11,
        fontweight="bold",
        color="#ffffff",
        y=0.97,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Adversarial sample grid saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    """
    Full robustness evaluation pipeline:
    clean accuracy → FGSM attack → C&W attack → visualisation → summary.
    """
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plot_path = logs_dir / "adversarial_samples.png"

    # ---- Device ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Hardware device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s  |  VRAM: %.2f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
        )

    # ---- Model & data ----------------------------------------------------
    model = load_model(args.weights_path, args.num_classes, device)

    logger.info("Loading test data from '%s' …", args.data_dir)
    test_loader: DataLoader = get_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info("Test loader ready | batches=%d", len(test_loader))

    # ====================================================================
    # Step 1: Clean Accuracy
    # ====================================================================
    logger.info("=" * 60)
    logger.info("STEP 1 — Clean Accuracy")
    logger.info("=" * 60)

    clean_acc, snap_images, snap_labels = evaluate_clean(model, test_loader, device)
    logger.info("Clean Accuracy: %.2f%%", clean_acc)

    # Predict on the snapshot batch for the visualisation row labels
    with torch.no_grad():
        snap_clean_logits = model(snap_images[: args.vis_samples].to(device))
        snap_clean_preds  = snap_clean_logits.argmax(dim=1).cpu()

    # ====================================================================
    # Step 2: FGSM Attack
    # FGSM perturbs every pixel by ε × sign(∇_x L).  One backward pass,
    # O(N) cost – fast but produces detectable structured noise patterns.
    # ====================================================================
    logger.info("=" * 60)
    logger.info(
        "STEP 2 — FGSM Attack  (ε=%.4f ≈ %d/255)",
        args.fgsm_eps,
        round(args.fgsm_eps * 255),
    )
    logger.info("=" * 60)

    fgsm_attack = torchattacks.FGSM(model, eps=args.fgsm_eps)

    fgsm_acc, fgsm_snap, fgsm_snap_preds = evaluate_adversarial(
        model       = model,
        loader      = test_loader,
        attack      = fgsm_attack,
        device      = device,
        phase_name  = "FGSM",
        snapshot_images = snap_images[: args.vis_samples],
    )
    logger.info("FGSM Robust Accuracy: %.2f%%", fgsm_acc)

    # ====================================================================
    # Step 3: Carlini & Wagner (C&W) L2 Attack
    # C&W solves a constrained optimisation problem over many Adam steps,
    # minimising ‖δ‖₂ while guaranteeing misclassification.  Far stronger
    # and stealthier than FGSM; bypasses most gradient-masking defences.
    # ====================================================================
    logger.info("=" * 60)
    logger.info(
        "STEP 3 — C&W L2 Attack  (c=%.1f, steps=%d)",
        args.cw_c,
        args.cw_steps,
    )
    logger.info("=" * 60)

    cw_attack = torchattacks.CW(model, c=args.cw_c, steps=args.cw_steps)

    cw_acc, cw_snap, cw_snap_preds = evaluate_adversarial(
        model       = model,
        loader      = test_loader,
        attack      = cw_attack,
        device      = device,
        phase_name  = "C&W ",
        snapshot_images = snap_images[: args.vis_samples],
    )
    logger.info("C&W Robust Accuracy: %.2f%%", cw_acc)

    # ====================================================================
    # Step 4: Visualisation
    # ====================================================================
    logger.info("=" * 60)
    logger.info("STEP 4 — Generating adversarial visualisation grid …")
    logger.info("=" * 60)

    save_adversarial_grid(
        clean_images = snap_images[: args.vis_samples],
        clean_labels = snap_labels[: args.vis_samples],
        fgsm_images  = fgsm_snap[: args.vis_samples],
        fgsm_preds   = fgsm_snap_preds[: args.vis_samples],
        cw_images    = cw_snap[: args.vis_samples],
        cw_preds     = cw_snap_preds[: args.vis_samples],
        clean_preds  = snap_clean_preds[: args.vis_samples],
        n_samples    = args.vis_samples,
        num_classes  = args.num_classes,
        save_path    = plot_path,
    )

    # ====================================================================
    # Summary report
    # ====================================================================
    acc_drop_fgsm = clean_acc - fgsm_acc
    acc_drop_cw   = clean_acc - cw_acc

    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║          AI-SOC Red Team — Robustness Evaluation Report      ║
╠══════════════════════════════════════════════════════════════╣
║  Model          : TrafficSignNet (baseline_cnn.pth)          ║
║  Dataset        : {args.data_dir:<44s}║
║  Device         : {str(device):<44s}║
╠══════════════════════════════════════════════════════════════╣
║  Clean Accuracy            :  {clean_acc:>6.2f}%                    ║
║                                                              ║
║  FGSM Robust Accuracy      :  {fgsm_acc:>6.2f}%  (↓ {acc_drop_fgsm:>5.2f}%)           ║
║    ε = {args.fgsm_eps:.4f} ({round(args.fgsm_eps*255)}/255)                                     ║
║                                                              ║
║  C&W  Robust Accuracy      :  {cw_acc:>6.2f}%  (↓ {acc_drop_cw:>5.2f}%)           ║
║    c = {args.cw_c:.1f},  steps = {args.cw_steps:<35d}║
╠══════════════════════════════════════════════════════════════╣
║  Adversarial grid → {str(plot_path):<41s}║
╚══════════════════════════════════════════════════════════════╝"""
    logger.info(summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)

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
1. Clean Accuracy       – baseline performance on unperturbed test set.
2. FGSM Robust Accuracy – performance after fast single-step attack.
3. C&W  Robust Accuracy – performance after strong optimisation attack.
4. Adversarial image grid  → logs/adversarial_samples.png
5. Accuracy drop chart     → logs/accuracy_drop.png

Usage (Kaggle / Cloud GPU):
    python -m src.attacks.evaluate_robustness \
        --data_dir   data/processed \
        --weights_path weights/baseline_cnn.pth \
        --batch_size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # headless – no display server on Kaggle / SSH
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
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
# SOC colour palette – shared across both visualisations for consistency
# ---------------------------------------------------------------------------

SOC_BG      = "#0f0f14"   # near-black canvas
SOC_SURFACE = "#1a1a24"   # slightly lighter panel surface
SOC_GRID    = "#2a2a3a"   # subtle grid / border lines
SOC_TEXT    = "#e0e0e0"   # primary labels
SOC_SUBTEXT = "#9e9e9e"   # secondary / footnote text
SOC_GREEN   = "#4CAF50"   # clean / safe
SOC_ORANGE  = "#FF9800"   # moderate threat (FGSM)
SOC_RED     = "#F44336"   # critical threat (C&W)
SOC_ACCENT  = "#69f0ae"   # survived annotation
SOC_WARN    = "#ff5252"   # fooled annotation

# ---------------------------------------------------------------------------
# GTSRB class names (43 classes)
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
        help="C&W trade-off constant c (higher → stronger attack / larger perturbation).",
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
        help="Number of sample images in the visualisation grid (≤ batch_size).",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Directory where all visualisation artefacts are saved.",
    )
    args = parser.parse_args()
    args.vis_samples = min(args.vis_samples, args.batch_size)
    return args


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    weights_path: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Instantiate TrafficSignNet and load pre-trained weights safely.

    ``map_location`` prevents GPU→CPU device-mismatch errors when the
    weights were saved on a different hardware configuration.

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
    state_dict = torch.load(str(path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Model loaded from '%s' on device '%s'.", path, device)
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluate clean (unperturbed) accuracy and capture the first batch for
    the visualisation grid.

    Returns
    -------
    Tuple[float, Tensor, Tensor]
        (accuracy_percent, first_batch_images_cpu, first_batch_labels_cpu)
    """
    correct: int = 0
    total:   int = 0
    snapshot_images: Optional[torch.Tensor] = None
    snapshot_labels: Optional[torch.Tensor] = None

    for batch_idx, (images, labels) in enumerate(
        tqdm(loader, desc="[CLEAN] Evaluating", unit="batch", dynamic_ncols=True)
    ):
        images = images.to(device)
        labels = labels.to(device)

        preds   = model(images).argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total   += labels.size(0)

        if batch_idx == 0:
            snapshot_images = images.cpu()
            snapshot_labels = labels.cpu()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, snapshot_images, snapshot_labels   # type: ignore[return-value]


def evaluate_adversarial(
    model: nn.Module,
    loader: DataLoader,
    attack: torchattacks.Attack,
    device: torch.device,
    phase_name: str,
    snapshot_images: torch.Tensor,
    snapshot_labels: torch.Tensor,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Generate adversarial examples with ``attack`` and measure robust accuracy.

    torchattacks requires the autograd graph to be live during the attack
    call itself; ``torch.no_grad()`` is applied only around the subsequent
    clean forward pass used for accuracy measurement.

    Returns
    -------
    Tuple[float, Tensor, Tensor]
        (robust_accuracy_percent,
         adversarial_snapshot_images_cpu,
         model_predictions_on_snapshot_cpu)
    """
    correct: int = 0
    total:   int = 0
    adv_snapshot: Optional[torch.Tensor] = None
    adv_preds:    Optional[torch.Tensor] = None

    for batch_idx, (images, labels) in enumerate(
        tqdm(loader, desc=f"[{phase_name}] Attacking", unit="batch", dynamic_ncols=True)
    ):
        images = images.to(device)
        labels = labels.to(device)

        # Attack call – autograd must be active here
        adv_images: torch.Tensor = attack(images, labels)

        with torch.no_grad():
            preds   = model(adv_images).argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total   += labels.size(0)

        # Perturb the fixed snapshot batch for the visualisation grid
        if batch_idx == 0:
            adv_first = attack(
                snapshot_images.to(device),
                snapshot_labels.to(device),
            )
            with torch.no_grad():
                snap_preds = model(adv_first).argmax(dim=1)
            adv_snapshot = adv_first.cpu()
            adv_preds    = snap_preds.cpu()

    robust_accuracy = 100.0 * correct / total if total > 0 else 0.0
    return robust_accuracy, adv_snapshot, adv_preds   # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Visualisation 1 – Adversarial image grid
# ---------------------------------------------------------------------------

def _tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised CHW tensor to a displayable HWC uint8 numpy array.
    Per-image min-max scaling handles any normalisation scheme without
    requiring knowledge of the original mean / std.
    """
    img = tensor.detach().float()
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo + 1e-8)
    return (np.clip(img.permute(1, 2, 0).numpy(), 0.0, 1.0) * 255).astype(np.uint8)


def save_adversarial_grid(
    clean_images: torch.Tensor,
    clean_labels: torch.Tensor,
    clean_preds:  torch.Tensor,
    fgsm_images:  torch.Tensor,
    fgsm_preds:   torch.Tensor,
    cw_images:    torch.Tensor,
    cw_preds:     torch.Tensor,
    n_samples:    int,
    num_classes:  int,
    save_path:    Path,
) -> None:
    """
    Render a three-row (Clean / FGSM / C&W) side-by-side comparison grid.

    Title colour convention:
        Green  (SOC_ACCENT) → model prediction correct (survived attack).
        Red    (SOC_WARN)   → model was fooled (misclassified).
    """
    n = min(n_samples, clean_images.size(0))

    fig = plt.figure(figsize=(n * 2.4, 3 * 3.0))
    fig.patch.set_facecolor(SOC_BG)

    gs = gridspec.GridSpec(
        3, n, figure=fig,
        hspace=0.55, wspace=0.08,
        left=0.07, right=0.97, top=0.88, bottom=0.04,
    )

    row_labels = ["Clean", "FGSM\n(ε=8/255)", "C&W\n(L2, 50 steps)"]
    for row_idx, row_label in enumerate(row_labels):
        fig.text(
            0.01, 1.0 - (row_idx + 0.5) / 3, row_label,
            va="center", ha="left",
            fontsize=9, fontweight="bold", color=SOC_TEXT, rotation=90,
        )

    rows_data: List[Tuple[torch.Tensor, torch.Tensor, bool]] = [
        (clean_images, clean_preds, True),
        (fgsm_images,  fgsm_preds,  False),
        (cw_images,    cw_preds,    False),
    ]

    for row_idx, (images, preds, is_clean_row) in enumerate(rows_data):
        for col_idx in range(n):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor(SOC_BG)
            ax.imshow(_tensor_to_display(images[col_idx]), interpolation="nearest")
            ax.axis("off")

            pred_idx = int(preds[col_idx].item())
            gt_idx   = int(clean_labels[col_idx].item())
            correct  = pred_idx == gt_idx

            if is_clean_row:
                title = f"GT: {label_name(gt_idx, num_classes)}\n→ {label_name(pred_idx, num_classes)}"
            else:
                title = f"→ {label_name(pred_idx, num_classes)}"

            ax.set_title(
                title,
                fontsize=6.5,
                color=SOC_ACCENT if correct else SOC_WARN,
                pad=3,
                fontweight="bold",
            )

    fig.suptitle(
        "AI-SOC Red Team — Adversarial Attack Visualisation\n"
        "Green = model survived  |  Red = model fooled",
        fontsize=11, fontweight="bold", color=SOC_TEXT, y=0.97,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Adversarial sample grid saved → %s", save_path)


# ---------------------------------------------------------------------------
# Visualisation 2 – Accuracy drop bar chart  (NEW)
# ---------------------------------------------------------------------------

def save_accuracy_chart(
    clean_acc: float,
    fgsm_acc:  float,
    cw_acc:    float,
    save_path: Path,
) -> None:
    """
    Render a SOC-styled bar chart comparing clean vs adversarial accuracies.

    Each bar is annotated with:
      • Its exact accuracy value at the bar top (e.g. ``98.95%``).
      • An absolute drop badge centred inside the bar for attacked
        variants (e.g. ``↓ 39.79%``), drawn as a rounded rectangle with
        a coloured border so it is immediately legible against the bar fill.

    Styling decisions
    -----------------
    * Dark ``SOC_BG`` / ``SOC_SURFACE`` palette matches the image grid.
    * Green / Orange / Red bars signal severity at a glance.
    * A subtle wider semi-transparent bar behind each main bar creates a
      soft glow without requiring external rendering dependencies.
    * Y-axis runs 0 → 110 to guarantee space above the tallest bar for
      the accuracy annotation without clipping.

    Parameters
    ----------
    clean_acc : Clean test-set accuracy (%).
    fgsm_acc  : Robust accuracy after FGSM attack (%).
    cw_acc    : Robust accuracy after C&W attack (%).
    save_path : Destination PNG path (parent dirs created if absent).
    """
    labels:     List[str]   = ["Clean\n(No Attack)", "FGSM\n(ε = 8/255)", "C&W\n(L2, 50 steps)"]
    accuracies: List[float] = [clean_acc, fgsm_acc, cw_acc]
    colours:    List[str]   = [SOC_GREEN, SOC_ORANGE, SOC_RED]
    drops:      List[float] = [0.0, clean_acc - fgsm_acc, clean_acc - cw_acc]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SOC_BG)
    ax.set_facecolor(SOC_SURFACE)

    # Horizontal grid drawn first so bars render on top
    ax.yaxis.grid(True, color=SOC_GRID, linewidth=0.8, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    x_pos      = np.arange(len(labels))
    bar_width  = 0.45

    # Primary bars
    bars = ax.bar(
        x_pos, accuracies,
        width=bar_width,
        color=colours,
        edgecolor=SOC_BG,
        linewidth=1.2,
        zorder=3,
    )

    # Soft glow: wider low-alpha bar behind each primary bar
    for pos, acc, col in zip(x_pos, accuracies, colours):
        ax.bar(pos, acc, width=bar_width + 0.10,
               color=col, alpha=0.15, zorder=2)

    # ---- Accuracy value annotations at bar tops --------------------------
    for bar, acc in zip(bars, accuracies):
        cx = bar.get_x() + bar.get_width() / 2
        ax.text(
            cx, bar.get_height() + 1.4,
            f"{acc:.2f}%",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            color=SOC_TEXT,
            path_effects=[pe.withStroke(linewidth=2, foreground=SOC_BG)],
            zorder=5,
        )

    # ---- Drop badge annotations centred inside attacked bars -------------
    for bar, drop, col in zip(bars[1:], drops[1:], colours[1:]):
        if drop <= 0.0:
            continue

        cx       = bar.get_x() + bar.get_width() / 2
        badge_cy = bar.get_height() / 2   # vertical midpoint of the bar
        badge_w  = 0.38
        badge_h  = 5.5                    # in data units (percentage points)

        # Rounded-rectangle badge background
        badge = mpatches.FancyBboxPatch(
            xy=(cx - badge_w / 2, badge_cy - badge_h / 2),
            width=badge_w,
            height=badge_h,
            boxstyle="round,pad=0.02",
            facecolor=SOC_BG,
            edgecolor=col,
            linewidth=1.4,
            alpha=0.90,
            zorder=6,
            transform=ax.transData,
        )
        ax.add_patch(badge)

        ax.text(
            cx, badge_cy,
            f"↓ {drop:.2f}%",
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            color=col,
            zorder=7,
        )

    # ---- Axes -----------------------------------------------------------
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 110)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11, color=SOC_TEXT)
    ax.set_ylabel("Accuracy (%)", fontsize=11, color=SOC_TEXT, labelpad=10)
    ax.set_xlabel("Attack Method", fontsize=11, color=SOC_TEXT, labelpad=10)
    ax.tick_params(axis="y", colors=SOC_SUBTEXT, labelsize=9)
    ax.tick_params(axis="x", colors=SOC_TEXT, length=0)

    # Show only the bottom spine as a thin separator line
    for name, spine in ax.spines.items():
        if name == "bottom":
            spine.set_color(SOC_GRID)
            spine.set_linewidth(0.8)
        else:
            spine.set_visible(False)

    # ---- Legend ----------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor=SOC_GREEN,  label="Clean  — no adversarial perturbation"),
        mpatches.Patch(facecolor=SOC_ORANGE, label="FGSM   — fast single-step L∞ attack"),
        mpatches.Patch(facecolor=SOC_RED,    label="C&W    — optimisation-based L2 attack"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        framealpha=0.25,
        facecolor=SOC_BG,
        edgecolor=SOC_GRID,
        labelcolor=SOC_TEXT,
    )

    # ---- Titles ----------------------------------------------------------
    ax.set_title(
        "AI-SOC Red Team — Model Robustness Under Adversarial Attack",
        fontsize=13, fontweight="bold",
        color=SOC_TEXT, pad=16,
    )
    fig.text(
        0.5, 0.01,
        "TrafficSignNet (baseline_cnn.pth)  |  GTSRB Test Set",
        ha="center", fontsize=8.5, color=SOC_SUBTEXT,
    )

    # ---- Save ------------------------------------------------------------
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Accuracy drop chart saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    """
    Full robustness evaluation pipeline:
    clean → FGSM → C&W → adversarial grid → accuracy chart → summary.
    """
    logs_dir   = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    grid_path  = logs_dir / "adversarial_samples.png"
    chart_path = logs_dir / "accuracy_drop.png"

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

    vis_images = snap_images[: args.vis_samples]
    vis_labels = snap_labels[: args.vis_samples]

    with torch.no_grad():
        snap_clean_preds = model(vis_images.to(device)).argmax(dim=1).cpu()

    # ====================================================================
    # Step 2: FGSM Attack
    # Single-step L∞ attack – fast, coarse, high throughput.
    # ====================================================================
    logger.info("=" * 60)
    logger.info(
        "STEP 2 — FGSM Attack  (ε=%.4f ≈ %d/255)",
        args.fgsm_eps, round(args.fgsm_eps * 255),
    )
    logger.info("=" * 60)

    fgsm_attack = torchattacks.FGSM(model, eps=args.fgsm_eps)
    fgsm_acc, fgsm_snap, fgsm_snap_preds = evaluate_adversarial(
        model=model, loader=test_loader, attack=fgsm_attack,
        device=device, phase_name="FGSM",
        snapshot_images=vis_images, snapshot_labels=vis_labels,
    )
    logger.info("FGSM Robust Accuracy: %.2f%%", fgsm_acc)

    # ====================================================================
    # Step 3: C&W L2 Attack
    # Iterative optimisation – stealthy, strong, bypasses many defences.
    # ====================================================================
    logger.info("=" * 60)
    logger.info(
        "STEP 3 — C&W L2 Attack  (c=%.1f, steps=%d)",
        args.cw_c, args.cw_steps,
    )
    logger.info("=" * 60)

    cw_attack = torchattacks.CW(model, c=args.cw_c, steps=args.cw_steps)
    cw_acc, cw_snap, cw_snap_preds = evaluate_adversarial(
        model=model, loader=test_loader, attack=cw_attack,
        device=device, phase_name="C&W ",
        snapshot_images=vis_images, snapshot_labels=vis_labels,
    )
    logger.info("C&W Robust Accuracy: %.2f%%", cw_acc)

    # ====================================================================
    # Step 4: Adversarial image grid
    # ====================================================================
    logger.info("=" * 60)
    logger.info("STEP 4 — Generating adversarial image grid …")
    logger.info("=" * 60)

    save_adversarial_grid(
        clean_images=vis_images,
        clean_labels=vis_labels,
        clean_preds=snap_clean_preds,
        fgsm_images=fgsm_snap[: args.vis_samples],
        fgsm_preds=fgsm_snap_preds[: args.vis_samples],
        cw_images=cw_snap[: args.vis_samples],
        cw_preds=cw_snap_preds[: args.vis_samples],
        n_samples=args.vis_samples,
        num_classes=args.num_classes,
        save_path=grid_path,
    )

    # ====================================================================
    # Step 5: Accuracy drop bar chart
    # ====================================================================
    logger.info("=" * 60)
    logger.info("STEP 5 — Generating accuracy drop chart …")
    logger.info("=" * 60)

    save_accuracy_chart(
        clean_acc=clean_acc,
        fgsm_acc=fgsm_acc,
        cw_acc=cw_acc,
        save_path=chart_path,
    )

    # ====================================================================
    # Terminal summary
    # ====================================================================
    drop_fgsm = clean_acc - fgsm_acc
    drop_cw   = clean_acc - cw_acc

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
║  FGSM Robust Accuracy      :  {fgsm_acc:>6.2f}%  (↓ {drop_fgsm:>5.2f}%)           ║
║    ε = {args.fgsm_eps:.4f} ({round(args.fgsm_eps*255)}/255)                                     ║
║                                                              ║
║  C&W  Robust Accuracy      :  {cw_acc:>6.2f}%  (↓ {drop_cw:>5.2f}%)           ║
║    c = {args.cw_c:.1f},  steps = {args.cw_steps:<35d}║
╠══════════════════════════════════════════════════════════════╣
║  Adversarial grid  → {str(grid_path):<40s}║
║  Accuracy chart    → {str(chart_path):<40s}║
╚══════════════════════════════════════════════════════════════╝"""
    logger.info(summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)

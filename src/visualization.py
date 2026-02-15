# ==================================================
# VISUALIZATION - Sonuc Gorsellestirme
# ==================================================

import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_color_palette(
    dominant_colors: List[dict],
    color_names: List[dict],
    output_dir: str,
) -> str:
    """Dominant renklerin paletini yatay bloklar halinde cizer.

    Her blok bir kume merkezinin rengini gosterir.
    Altinda RGB degeri, renk ismi ve yuzdesi yazar.

    Args:
        dominant_colors: get_dominant_colors() ciktisi.
        color_names: categorize_centers() ciktisi.
        output_dir: Cikti klasoru.

    Returns:
        Kaydedilen dosyanin tam yolu.
    """
    k = len(dominant_colors)

    fig, axes = plt.subplots(1, k, figsize=(2 * k, 3))
    fig.suptitle("Dominant Renk Paleti", fontsize=14, fontweight="bold")

    # color_names'i color_id'ye gore hizli erisim icin dict'e cevir
    name_map = {c["color_id"]: c["name"] for c in color_names}

    for i, color_info in enumerate(dominant_colors):
        ax = axes[i] if k > 1 else axes

        rgb = color_info["rgb"]
        color_id = color_info["color_id"]
        percentage = color_info["percentage"]
        name = name_map.get(color_id, "?")

        # Renk blogu olustur (0-1 araligina normalize et)
        color_block = np.array([[rgb]], dtype=np.uint8)
        ax.imshow(color_block, aspect="auto")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"%{percentage}", fontsize=9)
        ax.set_xlabel(f"{name}\n({rgb[0]},{rgb[1]},{rgb[2]})", fontsize=7)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "color_palette.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"[OK] Renk paleti kaydedildi: {filepath}")
    return filepath


def plot_comparison(
    original: np.ndarray,
    segmented: np.ndarray,
    output_dir: str,
) -> str:
    """Orijinal ve segmented goruntuyu yan yana gosterir.

    Args:
        original: Orijinal RGB goruntu.
        segmented: Segmented RGB goruntu.
        output_dir: Cikti klasoru.

    Returns:
        Kaydedilen dosyanin tam yolu.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Orijinal vs Segmented", fontsize=14, fontweight="bold"
    )

    ax1.imshow(original)
    ax1.set_title("Orijinal")
    ax1.axis("off")

    ax2.imshow(segmented)
    ax2.set_title(f"Segmented (K={len(np.unique(segmented.reshape(-1, 3), axis=0))} renk)")
    ax2.axis("off")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "comparison.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"[OK] Karsilastirma kaydedildi: {filepath}")
    return filepath


def plot_summary(
    original: np.ndarray,
    segmented: np.ndarray,
    dominant_colors: List[dict],
    color_names: List[dict],
    output_dir: str,
) -> str:
    """Tum sonuclari tek bir panelde ozetler.

    Ust satir: Orijinal ve segmented goruntu.
    Alt satir: Renk paleti barchar seklinde.

    Args:
        original: Orijinal RGB goruntu.
        segmented: Segmented RGB goruntu.
        dominant_colors: get_dominant_colors() ciktisi.
        color_names: categorize_centers() ciktisi.
        output_dir: Cikti klasoru.

    Returns:
        Kaydedilen dosyanin tam yolu.
    """
    name_map = {c["color_id"]: c["name"] for c in color_names}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Vision Color Pipeline - Ozet", fontsize=16, fontweight="bold"
    )

    # Ust sol: Orijinal
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(original)
    ax1.set_title("Orijinal Goruntu")
    ax1.axis("off")

    # Ust sag: Segmented
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(segmented)
    ax2.set_title("Segmented Goruntu")
    ax2.axis("off")

    # Alt: Renk dagilimi bar chart
    ax3 = fig.add_subplot(2, 1, 2)

    labels = []
    percentages = []
    bar_colors = []

    for color_info in dominant_colors:
        rgb = color_info["rgb"]
        color_id = color_info["color_id"]
        name = name_map.get(color_id, "?")

        labels.append(f"{name}\n({rgb[0]},{rgb[1]},{rgb[2]})")
        percentages.append(color_info["percentage"])
        bar_colors.append([c / 255.0 for c in rgb])

    bars = ax3.bar(range(len(labels)), percentages, color=bar_colors,
                   edgecolor="black", linewidth=0.5)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylabel("Yuzde (%)")
    ax3.set_title("Dominant Renk Dagilimi")

    # Bar'larin ustune yuzde yaz
    for bar, pct in zip(bars, percentages):
        ax3.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"%{pct}", ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "summary.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"[OK] Ozet panel kaydedildi: {filepath}")
    return filepath

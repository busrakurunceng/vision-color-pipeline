# ==================================================
# HISTOGRAM - Renk Dağılımı Analizi
# ==================================================

import os

import numpy as np
import matplotlib.pyplot as plt


def plot_rgb_histogram(image: np.ndarray, output_dir: str) -> str:
    """Her RGB kanalının histogramını ayrı alt grafiklerde çizer.

    Görüntüdeki kırmızı, yeşil ve mavi piksel yoğunluklarının
    0-255 aralığındaki dağılımını 3 ayrı grafik olarak gösterir.

    Args:
        image: RGB formatında numpy dizisi (H, W, 3).
        output_dir: Çıktı klasörünün yolu.

    Returns:
        Kaydedilen dosyanın tam yolu.
    """
    channel_names = ["Red", "Green", "Blue"]
    channel_colors = ["red", "green", "blue"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("RGB Kanal Histogramları", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes):
        ax.hist(
            image[:, :, i].ravel(),
            bins=256,
            range=(0, 256),
            color=channel_colors[i],
            alpha=0.7,
        )
        ax.set_title(channel_names[i])
        ax.set_xlabel("Piksel Değeri (0-255)")
        ax.set_ylabel("Frekans")
        ax.set_xlim([0, 256])

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "histogram_rgb.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"[OK] RGB histogramı kaydedildi: {filepath}")
    return filepath


def plot_combined_histogram(image: np.ndarray, output_dir: str) -> str:
    """Üç renk kanalını tek grafikte üst üste çizer.

    R, G, B dağılımlarını aynı eksende göstererek
    kanallar arası karşılaştırmayı kolaylaştırır.

    Args:
        image: RGB formatında numpy dizisi (H, W, 3).
        output_dir: Çıktı klasörünün yolu.

    Returns:
        Kaydedilen dosyanın tam yolu.
    """
    channel_names = ["Red", "Green", "Blue"]
    channel_colors = ["red", "green", "blue"]

    plt.figure(figsize=(10, 5))
    plt.title("Birleşik RGB Histogramı", fontsize=14, fontweight="bold")

    for i in range(3):
        plt.hist(
            image[:, :, i].ravel(),
            bins=256,
            range=(0, 256),
            color=channel_colors[i],
            alpha=0.4,
            label=channel_names[i],
        )

    plt.xlabel("Piksel Değeri (0-255)")
    plt.ylabel("Frekans")
    plt.legend()
    plt.xlim([0, 256])
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "histogram_combined.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"[OK] Birleşik histogram kaydedildi: {filepath}")
    return filepath

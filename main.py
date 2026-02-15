# ==================================================
# MAIN - Pipeline Orkestrasyonu
# ==================================================
# Bu dosya sadece modulleri cagirarak pipeline'i yonetir.
# Tum is mantigi src/ icerisindeki modullerde bulunur.
# main.py = controller, src/ = logic

from config import IMAGE_PATH, OUTPUT_DIR, K_CLUSTERS, RANDOM_STATE

from src.image_io import load_image, save_image
from src.pixel_analysis import get_image_info, extract_pixels
from src.histogram import plot_rgb_histogram, plot_combined_histogram
from src.clustering import apply_kmeans, get_dominant_colors
from src.segmentation import segment_image, create_label_map
from src.color_categorization import categorize_centers
from src.visualization import plot_color_palette, plot_comparison, plot_summary


def main():
    """Vision Color Pipeline - Ana fonksiyon.

    Bir goruntunun piksel analizinden renk kumeleme ve
    segmentasyona kadar tum adimlari sirasiyla calistirir.
    """
    print("=" * 55)
    print("  VISION COLOR PIPELINE")
    print("=" * 55)

    # 1. Goruntu yukleme
    print("\n[ADIM 1] Goruntu yukleniyor...")
    image = load_image(IMAGE_PATH)

    # 2. Piksel analizi
    print("\n[ADIM 2] Piksel analizi yapiliyor...")
    get_image_info(image)
    pixels = extract_pixels(image)

    # 3. Histogram
    print("\n[ADIM 3] Histogramlar olusturuluyor...")
    plot_rgb_histogram(image, OUTPUT_DIR)
    plot_combined_histogram(image, OUTPUT_DIR)

    # 4. K-Means kumeleme
    print("\n[ADIM 4] K-Means kumeleme basliyor...")
    labels, centers = apply_kmeans(pixels, K_CLUSTERS, RANDOM_STATE)
    dominant_colors = get_dominant_colors(centers, labels)

    # 5. Segmentasyon
    print("\n[ADIM 5] Segmentasyon yapiliyor...")
    segmented = segment_image(labels, centers, image.shape)
    create_label_map(labels, image.shape)
    save_image(segmented, "segmented.png", OUTPUT_DIR)

    # 6. Renk kategorizasyonu
    print("\n[ADIM 6] Renkler isimlendiriliyor...")
    color_names = categorize_centers(centers)

    # 7. Gorsellestirme
    print("\n[ADIM 7] Gorsellestirmeler olusturuluyor...")
    plot_color_palette(dominant_colors, color_names, OUTPUT_DIR)
    plot_comparison(image, segmented, OUTPUT_DIR)
    plot_summary(image, segmented, dominant_colors, color_names, OUTPUT_DIR)

    # Tamamlandi
    print("\n" + "=" * 55)
    print("  PIPELINE TAMAMLANDI!")
    print(f"  Ciktilar: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()

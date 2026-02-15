# ==================================================
# COLOR CATEGORIZATION - Renk Isimlendirme
# ==================================================
# RGB degerlerini insan tarafindan anlasilir
# renk isimlerine donusturur.
# HSV uzayindaki Hue (ton) degerine gore siniflandirir.

from typing import List

import cv2
import numpy as np


def rgb_to_hsv(r: int, g: int, b: int) -> tuple:
    """Tek bir RGB degerini HSV'ye cevirir.

    OpenCV'nin HSV skalasi:
      H: 0-179 (derece / 2)
      S: 0-255
      V: 0-255

    Args:
        r: Kirmizi kanal (0-255).
        g: Yesil kanal (0-255).
        b: Mavi kanal (0-255).

    Returns:
        (h, s, v) tuple'i.
    """
    pixel = np.uint8([[[r, g, b]]])
    hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)

    h, s, v = hsv_pixel[0][0]
    return int(h), int(s), int(v)


def classify_color(r: int, g: int, b: int) -> str:
    """Bir RGB degerini insan tarafindan anlasilir renk ismine donusturur.

    Oncelikle achromatic (renksiz) tonlari kontrol eder:
    siyah, beyaz, gri. Sonra HSV Hue degerine gore
    kromatik renkleri siniflandirir.

    Args:
        r: Kirmizi kanal (0-255).
        g: Yesil kanal (0-255).
        b: Mavi kanal (0-255).

    Returns:
        Renk ismi (ornegin: "Kirmizi", "Mavi", "Koyu Yesil").
    """
    h, s, v = rgb_to_hsv(r, g, b)

    # --- Achromatic (renksiz) tonlar ---
    if v < 40:
        return "Siyah"
    if s < 30 and v > 200:
        return "Beyaz"
    if s < 40:
        if v < 120:
            return "Koyu Gri"
        return "Acik Gri"

    # --- Chromatic (renkli) tonlar ---
    # Hue bazli siniflandirma (OpenCV: 0-179)
    brightness = "Koyu" if v < 120 else "Acik" if v > 200 else ""

    if h < 10 or h >= 170:
        color_name = "Kirmizi"
    elif h < 22:
        color_name = "Turuncu"
    elif h < 35:
        color_name = "Sari"
    elif h < 78:
        color_name = "Yesil"
    elif h < 105:
        color_name = "Mavi"
    elif h < 135:
        color_name = "Lacivert"
    elif h < 155:
        color_name = "Mor"
    else:
        color_name = "Pembe"

    if brightness:
        return f"{brightness} {color_name}"
    return color_name


def categorize_centers(centers: np.ndarray) -> List[dict]:
    """Tum kume merkezlerini isimlendirip listeler.

    Her kume merkezi icin RGB, HSV ve renk ismi bilgisini
    bir sozluk olarak dondurur.

    Args:
        centers: Kume merkezleri (K, 3) - RGB degerleri.

    Returns:
        Her renk icin sozluk listesi:
        [{"color_id": 0, "rgb": (R,G,B), "hsv": (H,S,V), "name": "..."}, ...]
    """
    results = []

    print("=" * 50)
    print("RENK KATEGORIZASYONU")
    print("=" * 50)

    for i, center in enumerate(centers):
        r, g, b = int(center[0]), int(center[1]), int(center[2])
        h, s, v = rgb_to_hsv(r, g, b)
        name = classify_color(r, g, b)

        results.append({
            "color_id": i,
            "rgb": (r, g, b),
            "hsv": (h, s, v),
            "name": name,
        })

        print(f"  Renk {i}: RGB({r:3d}, {g:3d}, {b:3d}) | "
              f"HSV({h:3d}, {s:3d}, {v:3d}) | {name}")

    print("=" * 50)

    return results

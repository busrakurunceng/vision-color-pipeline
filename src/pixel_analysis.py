# ==================================================
# PIXEL ANALYSIS - Piksel Matris Analizi
# ==================================================

import numpy as np


def get_image_info(image: np.ndarray) -> dict:
    """Görüntünün temel bilgilerini döndürür.

    Görüntünün aslında sayılardan oluşan bir matris olduğunu
    somut olarak gösterir: boyut, kanal sayısı, piksel aralığı,
    toplam piksel sayısı gibi bilgileri bir sözlük olarak verir.

    Args:
        image: RGB formatında numpy dizisi (H, W, 3).

    Returns:
        Görüntü bilgilerini içeren sözlük.
    """
    height, width, channels = image.shape

    info = {
        "height": height,
        "width": width,
        "channels": channels,
        "total_pixels": height * width,
        "dtype": str(image.dtype),
        "min_value": int(image.min()),
        "max_value": int(image.max()),
        "mean_value": round(float(image.mean()), 2),
    }

    print("=" * 40)
    print("GÖRÜNTÜ BİLGİLERİ")
    print("=" * 40)
    print(f"  Boyut       : {width} x {height}")
    print(f"  Kanal sayısı: {channels} (RGB)")
    print(f"  Toplam piksel: {info['total_pixels']:,}")
    print(f"  Veri tipi   : {info['dtype']}")
    print(f"  Piksel aralığı: [{info['min_value']}, {info['max_value']}]")
    print(f"  Ortalama değer: {info['mean_value']}")
    print("=" * 40)

    return info


def extract_pixels(image: np.ndarray) -> np.ndarray:
    """Görüntüyü 2D piksel matrisine dönüştürür.

    (H, W, 3) boyutundaki görüntüyü (H*W, 3) boyutuna
    düzleştirir. Bu format K-Means gibi algoritmaların
    beklediği girdi formatıdır.

    Her satır bir pikselin [R, G, B] değerlerini temsil eder.

    Args:
        image: RGB formatında numpy dizisi (H, W, 3).

    Returns:
        (H*W, 3) boyutunda 2D numpy dizisi (float32).
    """
    height, width, channels = image.shape

    # (H, W, 3) -> (H*W, 3) düzleştirme
    pixels = image.reshape(-1, channels)

    # K-Means float bekler, uint8'den çeviriyoruz
    pixels = pixels.astype(np.float32)

    print(f"[OK] Piksel matrisi oluşturuldu: {pixels.shape}")
    print(f"     Orijinal: ({height}, {width}, {channels})")
    print(f"     Düzleştirilmiş: ({pixels.shape[0]}, {pixels.shape[1]})")
    print(f"     İlk 3 piksel (RGB): {pixels[:3].astype(int).tolist()}")

    return pixels

from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_image_features(image_path: Path) -> dict[str, Any]:
    features: dict[str, Any] = {
        "image_file_size": image_path.stat().st_size if image_path.exists() else 0,
    }
    if not image_path.exists() or not image_path.is_file():
        return features

    try:
        from PIL import Image, ImageOps, ImageStat
    except ImportError:
        features["image_feature_status"] = "pillow_missing"
        return features

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            gray = ImageOps.grayscale(image)
            sample = gray.resize((min(width, 64), min(height, 64)))
            pixel_data = (
                sample.get_flattened_data()
                if hasattr(sample, "get_flattened_data")
                else sample.getdata()
            )
            pixels = list(pixel_data)
            stat = ImageStat.Stat(sample)
            mean = float(stat.mean[0])
            stddev = float(stat.stddev[0])
            features.update(
                {
                    "image_width": width,
                    "image_height": height,
                    "image_aspect_ratio": _safe_div(width, height),
                    "image_megapixels": (width * height) / 1_000_000,
                    "image_brightness": mean / 255,
                    "image_contrast": stddev / 255,
                    "image_sharpness_proxy": _sharpness_proxy(pixels, sample.size[0]),
                    "image_feature_status": "ok",
                }
            )
    except Exception:
        features["image_feature_status"] = "error"
    return features


def _sharpness_proxy(pixels: list[int], width: int) -> float:
    if len(pixels) < 2 or width < 2:
        return 0.0
    diffs: list[int] = []
    for index, value in enumerate(pixels):
        if index % width != width - 1:
            diffs.append(abs(value - pixels[index + 1]))
        if index + width < len(pixels):
            diffs.append(abs(value - pixels[index + width]))
    return _safe_div(sum(diffs), len(diffs)) / 255


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

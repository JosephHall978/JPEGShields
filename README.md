# JPeG Shields

> **JPEG Trust Watermarking Benchmark | ICIP 2026 Grand Challenge**

![JPeG Shields Logo](logo__1_.png)

JPeG Shields is a frequency-domain image watermarking system submitted to the **JPEG Trust Watermarking Grand Challenge at ICIP 2026**. The system embeds imperceptible 100-bit binary watermarks into images using the Discrete Fourier Transform (DFT) and is evaluated on imperceptibility and robustness against a comprehensive suite of 18 image attacks.

---

## Repository Structure

```
JPEGShields/
├── WaterMarker.py                              # Core watermarking class (embed, recover, evaluate)
├── jpegshield_pipeline.ipynb                   # End-to-end evaluation pipeline (Google Colab)
├── jpegshield_pipeline_results_summary.csv     # Aggregated results (mean ± std per attack)
└── README.md
```

---

## How It Works

### Watermark Embedding (`WaterMarker.generate`)

The watermark is embedded into each RGB channel independently:

1. Apply a 2D FFT with frequency shift (`fftshift`) to convert to the frequency domain.
2. Extract magnitude and phase components.
3. For each of the 100 watermark bits, modify a specific diagonal position in the magnitude spectrum — setting it to `mean_magnitude × tolerance_factor` for a `1` bit, or `0` for a `0` bit.
4. Reconstruct the image via inverse FFT and clip pixel values to `[0, 255]`.

### Watermark Recovery (`WaterMarker.recover`)

1. Transform the (possibly attacked) image back to the frequency domain.
2. Sample the same diagonal positions used during embedding.
3. Threshold each sampled value against the mean magnitude to recover a binary bit.
4. Average recovered sequences across all three channels and apply a majority vote.

---

## Files

### `WaterMarker.py`

The core class implementing:

| Method | Description |
|---|---|
| `generate(image, watermark)` | Embeds a 100-bit binary watermark via DFT magnitude modification |
| `recover(image)` | Extracts the recovered watermark from a (possibly attacked) image |
| `compute_psnr(orig, watermarked)` | Peak Signal-to-Noise Ratio |
| `compute_wpsnr(orig, watermarked)` | Weighted PSNR using luminance-based perceptual weighting |
| `compute_ssim(orig, watermarked)` | Structural Similarity Index |
| `compute_jnd(orig, watermarked)` | Just Noticeable Difference (mean absolute luminance difference) |
| `evaluate(image, original_watermark)` | Returns number of incorrectly recovered bits |
| `evaluate_watermarking(orig, watermarked)` | Returns a dict of all quality metrics |

**Key parameters:**

- `seq_len = 100` — watermark length in bits
- `tolerance_factor = 20` — multiplier on mean magnitude for embedding strength
- `threshold = 0.5` — recovery threshold divisor against mean magnitude

### `jpegshield_pipeline.ipynb`

A Google Colab notebook that orchestrates the full evaluation pipeline:

1. **Setup** — installs dependencies (`opencv-python`, `scikit-image`, `scipy`, `watermarkbench`, `pillow-jxl`, `ultralytics`)
2. **Data loading** — mounts Google Drive and scans `Camera_Capture` and `Synthetic` subfolders
3. **Watermark generation** — produces a reproducible 100-bit watermark (random seed `42`)
4. **Attack evaluation** — runs each image through all 18 attack types and records metrics
5. **Outputs saved to Google Drive:**
   - `outputs/watermarked/` — watermarked images
   - `outputs/processed/` — attacked images
   - `outputs/watermarks/` — ground truth and recovered bit strings
   - `jpegshield_pipeline_results.csv` — full per-image results
6. **Visualisation** — seaborn bar plots of mean metrics per attack type

---

## Setup & Usage

### Requirements

```bash
pip install opencv-python scikit-image scipy pandas tqdm matplotlib pillow-jxl ultralytics
pip install "git+https://github.com/JPEG-Trust-Community/watermarking.git#subdirectory=evaluation_metric/package"
```

### Running the Pipeline

The pipeline is designed for **Google Colab** with Google Drive mounted:

```bash
git clone https://github.com/JosephHall978/JPEGShields.git
```

Set the following paths at the top of `jpegshield_pipeline.ipynb`:

```python
DATASET_ROOT  = "/content/drive/MyDrive/JPEGShields-main/Watermark Evaluation Dataset-Public"
WATERMARKER_PATH = "/content/JPEGShields/WaterMarker.py"
OUTPUT_ROOT   = "/content/drive/MyDrive/JPEGShields-main/outputs"
```

> An OpenAI API key is required (stored as a Colab secret under `OPENAI_API_KEY`) for the AI-based attacks (`create_ai`, `replace_ai`, `remove_ai`).

### Standalone Usage

```python
import numpy as np
import cv2
from WaterMarker import WaterMarker

marker = WaterMarker()

img = cv2.cvtColor(cv2.imread("image.png"), cv2.COLOR_BGR2RGB)

rng = np.random.default_rng(42)
watermark = rng.integers(0, 2, size=100, dtype=np.uint8)

watermarked = marker.generate(img, watermark)
recovered   = marker.recover(watermarked)
metrics     = marker.evaluate_watermarking(img, watermarked)
ber         = float(np.mean(watermark != recovered))

print(metrics)
print("BER:", ber)
```

---

## Evaluation Metrics

| Metric | Description | Direction |
|---|---|---|
| PSNR | Standard pixel-level distortion measure | ↑ higher is better |
| wPSNR | Perceptually weighted PSNR (luminance-sensitive) | ↑ higher is better |
| SSIM | Structural similarity of luminance, contrast, and structure | ↑ higher is better |
| JND | Mean absolute luminance difference — perceptual visibility | ↓ lower is better |
| BER | Bit Error Rate — fraction of incorrectly recovered bits | ↓ lower is better |

---

## Results

Evaluation was performed on the JPEG Trust public benchmark dataset across `Camera_Capture` and `Synthetic` image subsets. Each image was watermarked with a fixed 100-bit sequence (seed `42`) and then subjected to each of the 18 attacks independently. Imperceptibility metrics (PSNR, wPSNR, SSIM, JND) are consistent across attacks as they measure watermark visibility on the clean watermarked image before any attack is applied.

### Imperceptibility (Watermarked vs. Original)

The watermark is perceptually invisible across all evaluated images:

| Metric | Mean | Std |
|---|---|---|
| PSNR (dB) | 47.46 | ±2.98 |
| wPSNR (dB) | 49.26 | ±3.29 |
| SSIM | 0.9919 | ±0.0060 |
| JND | 0.7354 | ±0.3178 |

### Robustness — BER per Attack

Lower BER = more bits recovered correctly. BER = 0.5 indicates complete watermark loss (random chance).

| Attack | BER Mean | BER Std | Notes |
|---|---|---|---|
| **none** (baseline) | **0.1042** | ±0.1680 | Strong recovery with no attack |
| **gaussian_noise** | **0.0150** | ±0.0288 | Best robustness — noise preserves DFT structure |
| **speckle_noise** | **0.0258** | ±0.0578 | Very robust against multiplicative noise |
| **brightness** | **0.0973** | ±0.1652 | Robust — linear scaling preserves magnitude ratios |
| **sharpness** | **0.1450** | ±0.1811 | Mostly robust |
| **jpeg2000** | **0.4116** | ±0.1577 | Partial degradation |
| **median_filtering** | **0.5034** | ±0.0703 | Near-total watermark loss |
| **blurring** | **0.5244** | ±0.0401 | Near-total watermark loss |
| **jpeg** | **0.5244** | ±0.0396 | Near-total watermark loss |
| **crop** | **0.5319** | ±0.0109 | Near-total watermark loss |
| **flipping** | **0.5200** | ±0.0000 | Complete watermark loss |
| **rotate** | **0.5300** | ±0.0000 | Complete watermark loss — geometric transforms break diagonal frequency positions |
| **scaled** | **0.5300** | ±0.0007 | Complete watermark loss |
| **create_ai** | N/A | — | Not evaluated (pipeline error) |
| **replace_ai** | N/A | — | Not evaluated (pipeline error) |
| **remove_ai** | N/A | — | Not evaluated (pipeline error) |
| **jpegai** | N/A | — | Not evaluated (pipeline error) |
| **jpegxl** | N/A | — | Not evaluated (pipeline error) |

### Summary

The DFT diagonal-encoding approach achieves strong imperceptibility (PSNR ~47.5 dB, SSIM ~0.992) and is robust against additive noise attacks (gaussian, speckle) and global tonal changes (brightness, sharpness). However, geometric transforms (rotate, scale, flip, crop) and spatial filtering (blur, median, JPEG) cause near-complete watermark loss, as they displace or destroy the specific diagonal frequency components used for encoding. These are known weaknesses of fixed-position frequency-domain watermarking and represent the primary areas for future improvement.

---

## Competition

This system is submitted to the **[JPEG Trust Watermarking Grand Challenge](https://jpeg.org/jpegai/watermarking)** at **ICIP 2026**. The challenge evaluates watermarking methods on imperceptibility (PSNR, wPSNR, SSIM, JND) and robustness (BER) across a standardised benchmark dataset.

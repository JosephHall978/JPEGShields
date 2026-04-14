# JPeG Shields

> **JPEG Trust Watermarking | ICIP 2026 Grand Challenge**

![JPeG Shields Logo](logo__1_.png)

JPeG Shields is a frequency-domain image watermarking system developed for the **JPEG Trust Watermarking Grand Challenge at ICIP 2026**. The system embeds imperceptible binary watermarks into images using the Discrete Fourier Transform (DFT) and evaluates their robustness against a comprehensive suite of image attacks.

---

## Repository Structure

```
JPEGShields/
├── WaterMarker.py               # Core watermarking class (embed, recover, evaluate)
├── jpegshield_pipeline.ipynb    # End-to-end evaluation pipeline (Google Colab)
└── README.md
```

---

## How It Works

### Watermark Embedding (`WaterMarker.generate`)

The watermark is embedded into each RGB channel independently using the following steps:

1. Apply a 2D FFT with frequency shift (`fftshift`) to move to the frequency domain.
2. Extract the magnitude and phase components.
3. For each of the 100 watermark bits, modify a specific diagonal position in the magnitude spectrum — setting it to a boosted value (`mean_magnitude × tolerance_factor`) for a `1` bit, or `0` for a `0` bit.
4. Reconstruct the image via inverse FFT and clip pixel values to `[0, 255]`.

### Watermark Recovery (`WaterMarker.recover`)

1. Transform the (potentially attacked) image back into the frequency domain.
2. Sample the same diagonal positions used during embedding.
3. Threshold each sampled value against the mean magnitude to recover a binary bit.
4. Average the recovered sequences across all three channels and apply a majority vote.

---

## Files

### `WaterMarker.py`

The core class implementing:

| Method | Description |
|---|---|
| `generate(image, watermark)` | Embeds a 100-bit binary watermark into an image using DFT magnitude modification |
| `recover(image)` | Extracts and returns the recovered watermark from a (possibly attacked) image |
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
2. **Data loading** — mounts Google Drive and scans `Camera_Capture` and `Synthetic` subfolders of the evaluation dataset
3. **Watermark generation** — produces a reproducible 100-bit watermark using a fixed random seed (`42`)
4. **Attack evaluation** — runs each image through all 18 attack types and measures robustness

The 18 attacks tested are:

| Attack | Parameters |
|---|---|
| `none` | No attack (baseline) |
| `rotate` | 15° rotation |
| `crop` | 10% crop |
| `scaled` | 0.5× downscale |
| `flipping` | Horizontal flip |
| `jpeg` | Quality 50 |
| `jpeg2000` | Rate 10 |
| `jpegai` | Rate 1 |
| `jpegxl` | Quality 12 |
| `gaussian_noise` | σ = 0.01 |
| `speckle_noise` | σ = 0.3 |
| `blurring` | Kernel size 5 |
| `brightness` | Factor 1.3 |
| `sharpness` | Factor 1.25 |
| `median_filtering` | Kernel size 5 |
| `create_ai` | AI-generated replacement |
| `replace_ai` | AI inpainting |
| `remove_ai` | AI watermark removal |

5. **Metrics computed per image per attack:**
   - `PSNR` — Peak Signal-to-Noise Ratio
   - `wPSNR` — Weighted PSNR
   - `SSIM` — Structural Similarity
   - `JND` — Just Noticeable Difference
   - `BER` — Bit Error Rate (primary robustness metric)

6. **Outputs saved to Google Drive:**
   - `outputs/watermarked/` — watermarked images
   - `outputs/processed/` — attacked images
   - `outputs/watermarks/` — ground truth and recovered bit strings
   - `jpegshield_pipeline_results.csv` — full results table

7. **Visualisation** — bar plots of mean metrics per attack type using seaborn

---

## Setup & Usage

### Requirements

```bash
pip install opencv-python scikit-image scipy pandas tqdm matplotlib pillow-jxl ultralytics
pip install "git+https://github.com/JPEG-Trust-Community/watermarking.git#subdirectory=evaluation_metric/package"
```

### Running the Pipeline

The pipeline is designed for **Google Colab** with Google Drive mounted. Clone the repository and open the notebook:

```bash
git clone https://github.com/JosephHall978/JPEGShields.git
```

Then set the following paths at the top of `jpegshield_pipeline.ipynb`:

```python
DATASET_ROOT = "/content/drive/MyDrive/JPEGShields-main/Watermark Evaluation Dataset-Public"
WATERMARKER_PATH = "/content/JPEGShields/WaterMarker.py"
OUTPUT_ROOT = "/content/drive/MyDrive/JPEGShields-main/outputs"
```

An OpenAI API key is required (stored as a Colab secret under `OPENAI_API_KEY`) for the AI-based attacks (`create_ai`, `replace_ai`, `remove_ai`).

### Standalone Usage

```python
import numpy as np
import cv2
from WaterMarker import WaterMarker

marker = WaterMarker()

# Load image
img = cv2.cvtColor(cv2.imread("image.png"), cv2.COLOR_BGR2RGB)

# Generate a watermark
rng = np.random.default_rng(42)
watermark = rng.integers(0, 2, size=100, dtype=np.uint8)

# Embed
watermarked = marker.generate(img, watermark)

# Recover
recovered = marker.recover(watermarked)

# Evaluate
metrics = marker.evaluate_watermarking(img, watermarked)
ber = np.mean(watermark != recovered)

print(metrics)
print("BER:", ber)
```

---

## Evaluation Metrics

| Metric | Description | Higher is better? |
|---|---|---|
| PSNR | Standard pixel-level distortion measure | ✅ |
| wPSNR | Perceptually weighted PSNR (luminance-sensitive) | ✅ |
| SSIM | Structural similarity of luminance, contrast, and structure | ✅ |
| JND | Mean absolute luminance difference — perceptual visibility of watermark | ❌ (lower = less visible) |
| BER | Fraction of incorrectly recovered bits after attack | ❌ (lower = more robust) |

---

## Results

*Results to be added following full evaluation on the ICIP 2026 dataset.*

---

## Competition

This system was developed for the **[JPEG Trust Watermarking Grand Challenge](https://jpeg.org/jpegai/watermarking)** at **ICIP 2026**. The challenge evaluates watermarking methods on imperceptibility (PSNR, wPSNR, SSIM, JND) and robustness (BER) across a standardised image dataset including camera captures and synthetic images.

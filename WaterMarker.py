import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr

class WaterMarker():
    def __init__(self):
        self.mask_size = 25
        self.seq_len = 100
        self.pad = 0
        if self.pad > 0:
            self.tolerance_factor = 25*(1/self.pad)
            self.threshold = 1*(1/self.pad)
        else:
            self.tolerance_factor = 20
            self.threshold = 0.5

    def generate(self, image, watermark):
        work_img = image.astype(np.float64)
        watermarked_layers = []
        for ch in range(work_img.shape[2]):
            dft = fftshift(fft2(work_img[:, :, ch]))
            mag = np.abs(dft)
            phase = np.angle(dft)

            mean_mag = np.mean(mag)
            for i in range(self.seq_len):
                mag[i+int(len(mag)*self.pad), i+int(len(mag)*self.pad)] = mean_mag*self.tolerance_factor if watermark[i] == 1 else 0

            new_dft = ifftshift(mag * np.exp(1j * phase))
            watermarked_layers.append(np.real(ifft2(new_dft)))

        return np.stack(watermarked_layers, axis=2).clip(0, 255).astype(np.uint8)

    def recover(self, image):
        work_img = image.astype(np.float64)
        sequences = []
        for ch in range(work_img.shape[2]):
            dft = fftshift(fft2(work_img[:, :, ch]))
            mag = np.abs(dft)

            extracted_values = []
            for i in range(self.seq_len):
                extracted_values.append(mag[i+int(len(mag)*self.pad), i+int(len(mag)*self.pad)])
            avg_signal = np.mean(mag)

            recovered_seq = [1 if v > avg_signal/self.threshold else 0 for v in extracted_values]
            sequences.append(recovered_seq)

        recover_watermark = np.mean(np.array(sequences), axis=0)
        recover_watermark = [1 if w > 0.5 else 0 for w in recover_watermark]
        return np.array(recover_watermark)

    def compute_psnr(self, orig, watermarked):
        return peak_signal_noise_ratio(orig, watermarked, data_range=255)

    def compute_wpsnr(self, orig, watermarked):
        y_orig = rgb2ycbcr(orig)[..., 0].astype(np.float64)
        y_wm = rgb2ycbcr(watermarked)[..., 0].astype(np.float64)
        diff2 = (y_orig - y_wm) ** 2
        w = 1.0 / (1.0 + y_orig / 255.0)
        num = (255.0 ** 2) * np.sum(w)
        den = np.sum(w * diff2) + 1e-12
        return 10 * np.log10(num / den)

    def compute_ssim(self, orig, watermarked):
        return structural_similarity(orig, watermarked, channel_axis=-1, data_range=255)

    def compute_jnd(self, orig, watermarked):
        y_orig = rgb2ycbcr(orig)[..., 0].astype(np.float64)
        y_wm = rgb2ycbcr(watermarked)[..., 0].astype(np.float64)
        return np.mean(np.abs(y_orig - y_wm))

    def evaluate(self, image, original_watermark):
        recovered = self.recover(image)
        bits_changed = np.sum(original_watermark != recovered)
        return bits_changed

    def evaluate_watermarking(self,original_img, watermarked_img):
        metrics = {}
        metrics["PSNR"] = self.compute_psnr(original_img, watermarked_img)
        metrics["wPSNR"] = self.compute_wpsnr(original_img, watermarked_img)
        metrics["SSIM"] = self.compute_ssim(original_img, watermarked_img)
        metrics["JND"] = self.compute_jnd(original_img, watermarked_img)
        return metrics

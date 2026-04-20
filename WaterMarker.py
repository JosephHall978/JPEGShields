import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr

class HammingECC:
    _P = np.array([
        [1,1,0,1],[1,0,1,1],[1,0,0,0],
        [0,1,1,1],[0,1,0,0],[0,0,1,0],[0,0,0,1],
        [1,1,1,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],
    ], dtype=np.uint8)  # shape (11,4)

    # Parity-check matrix H  (4×15)
    _H = np.array([
        [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
        [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
        [0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    ], dtype=np.uint8)

    K, N = 11, 15

    @classmethod
    def encode(cls, bits: np.ndarray) -> np.ndarray:
        pad = (-len(bits)) % cls.K
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        chunks = bits.reshape(-1, cls.K)
        parity = (chunks @ cls._P) % 2          # (n_chunks, 4)
        codewords = np.concatenate([chunks, parity], axis=1)  # (n_chunks, 15)
        return codewords.flatten(), pad

    @classmethod
    def decode(cls, bits: np.ndarray, pad: int) -> np.ndarray:
        bits = np.array(bits, dtype=np.uint8)
        n_words = len(bits) // cls.N
        codewords = bits[: n_words * cls.N].reshape(n_words, cls.N)
        decoded = []
        for cw in codewords:
            syndrome = (cls._H @ cw) % 2          # shape (4,)
            err_pos  = int(''.join(syndrome[::-1].astype(str)), 2) - 1
            if 0 <= err_pos < cls.N:
                cw = cw.copy()
                cw[err_pos] ^= 1
            decoded.append(cw[:cls.K])
        result = np.concatenate(decoded)
        return result[:len(result) - pad] if pad else result

class WaterMarker:
    def __init__(self):
        self.seq_len   = 100
        self.alpha     = 0.75
        self.block_dim = 4
        self.repeats   = 3

    def _encode_payload(self, watermark: np.ndarray):
        wm = np.array(watermark, dtype=np.uint8)
        encoded, self._ecc_pad = HammingECC.encode(wm)
        repeated = np.tile(encoded, self.repeats)
        self._total_bits = len(repeated)
        self._encoded_len = len(encoded)
        return repeated

    def _decode_payload(self, bits: np.ndarray) -> np.ndarray:
        bits = np.array(bits, dtype=np.float64)
        chunk = len(bits) // self.repeats
        votes = bits[:chunk * self.repeats].reshape(self.repeats, chunk)
        majority = (np.mean(votes, axis=0) > 0.5).astype(np.uint8)
        return HammingECC.decode(majority, self._ecc_pad)

    def _get_blocks(self, h, w, n_bits):
        bd = self.block_dim
        rows = np.arange(bd, h - bd, bd)
        cols = np.arange(bd, w - bd, bd)
        rng  = np.random.default_rng(seed=42)
        grid = [(r, c) for r in rows for c in cols]
        rng.shuffle(grid)
        if len(grid) < n_bits:
            raise RuntimeError(
                f"Image too small: need {n_bits} blocks of {bd}×{bd}, "
                f"only {len(grid)} fit."
            )
        return grid[:n_bits]

    @staticmethod
    def _embed_bit_svd(block: np.ndarray, bit: int, alpha: float) -> np.ndarray:
        U, S, Vt = np.linalg.svd(block, full_matrices=False)
        ref = S[1] if len(S) > 1 else S[0]
        if bit == 1:
            S[0] = ref * (1 + alpha * 8)
        else:
            S[0] = ref * (1 + alpha * 1)
        return U @ np.diag(S) @ Vt

    @staticmethod
    def _read_bit_svd(block: np.ndarray, alpha: float) -> float:
        U, S, Vt = np.linalg.svd(block, full_matrices=False)
        if len(S) < 2 or S[1] < 1e-10:
            return 0.5
        ratio = S[0] / S[1]
        mid   = 1 + alpha * 4.5          # midpoint between bit-0 and bit-1 levels
        return 1.0 if ratio > mid else 0.0

    def generate(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        work   = image.astype(np.float64)
        h, w   = work.shape[:2]
        bits   = self._encode_payload(watermark)
        blocks = self._get_blocks(h, w, len(bits))

        watermarked_layers = []
        for ch in range(work.shape[2]):
            channel = work[:, :, ch].copy()
            dft     = fftshift(fft2(channel))
            mag     = np.abs(dft)
            phase   = np.angle(dft)

            bd = self.block_dim
            for bit_idx, (r0, c0) in enumerate(blocks):
                blk          = mag[r0:r0+bd, c0:c0+bd].copy()
                mag[r0:r0+bd, c0:c0+bd] = self._embed_bit_svd(blk, bits[bit_idx], self.alpha)

            new_dft = ifftshift(mag * np.exp(1j * phase))
            channel_wm = np.real(ifft2(new_dft))
            watermarked_layers.append(channel_wm)

        return np.stack(watermarked_layers, axis=2).clip(0, 255).astype(np.uint8)

    def recover(self, image: np.ndarray) -> np.ndarray:
        work   = image.astype(np.float64)
        h, w   = work.shape[:2]
        blocks = self._get_blocks(h, w, self._total_bits)

        channel_bits = []
        for ch in range(work.shape[2]):
            channel = work[:, :, ch]
            dft     = fftshift(fft2(channel))
            mag     = np.abs(dft)

            bd   = self.block_dim
            soft = []
            for (r0, c0) in blocks:
                blk = mag[r0:r0+bd, c0:c0+bd]
                soft.append(self._read_bit_svd(blk, self.alpha))
            channel_bits.append(soft)

        # Average soft decisions across channels, then hard-decide
        avg  = np.mean(np.array(channel_bits), axis=0)
        hard = (avg > 0.5).astype(np.uint8)
        return self._decode_payload(hard)

    def compute_psnr(self, orig, watermarked):
        return peak_signal_noise_ratio(orig, watermarked, data_range=255)

    def compute_wpsnr(self, orig, watermarked):
        y_orig = rgb2ycbcr(orig)[..., 0].astype(np.float64)
        y_wm   = rgb2ycbcr(watermarked)[..., 0].astype(np.float64)
        diff2  = (y_orig - y_wm) ** 2
        w      = 1.0 / (1.0 + y_orig / 255.0)
        return 10 * np.log10((255.0 ** 2) * np.sum(w) / (np.sum(w * diff2) + 1e-12))

    def compute_ssim(self, orig, watermarked):
        return structural_similarity(orig, watermarked, channel_axis=-1, data_range=255)

    def compute_jnd(self, orig, watermarked):
        y_orig = rgb2ycbcr(orig)[..., 0].astype(np.float64)
        y_wm   = rgb2ycbcr(watermarked)[..., 0].astype(np.float64)
        return np.mean(np.abs(y_orig - y_wm))

    def evaluate(self, image, original_watermark):
        recovered = self.recover(image)
        n = min(len(original_watermark), len(recovered))
        return int(np.sum(original_watermark[:n] != recovered[:n]))

    def evaluate_watermarking(self, original_img, watermarked_img):
        return {
            "PSNR":  self.compute_psnr(original_img, watermarked_img),
            "wPSNR": self.compute_wpsnr(original_img, watermarked_img),
            "SSIM":  self.compute_ssim(original_img, watermarked_img),
            "JND":   self.compute_jnd(original_img, watermarked_img),
        }
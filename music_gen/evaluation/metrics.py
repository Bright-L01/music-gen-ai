"""
Audio quality evaluation metrics for music generation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from scipy import linalg
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class AudioQualityMetrics:
    """Comprehensive audio quality metrics for music generation evaluation."""

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 128,
        compute_fad: bool = True,
        compute_clap: bool = False,  # Requires CLAP model
        compute_inception_score: bool = True,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.compute_fad = compute_fad
        self.compute_clap = compute_clap
        self.compute_inception_score = compute_inception_score

        # Pre-compute mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0,
            fmax=sample_rate // 2,
        )

        # Initialize feature extractors
        if compute_fad:
            self._init_fad_model()

        if compute_clap:
            self._init_clap_model()

    def _init_fad_model(self):
        """Initialize VGGish model for FAD computation."""
        try:
            import torchvggish

            self.vggish_model = torchvggish.vggish()
            self.vggish_model.eval()
            logger.info("Initialized VGGish model for FAD computation")
        except ImportError:
            logger.warning("torchvggish not available, FAD computation disabled")
            self.compute_fad = False

    def _init_clap_model(self):
        """Initialize CLAP model for text-audio alignment."""
        try:
            # This would require a CLAP model implementation
            # For now, we'll skip this and use a placeholder
            logger.warning("CLAP model not implemented, using placeholder")
            self.compute_clap = False
        except Exception as e:
            logger.warning(f"Failed to initialize CLAP model: {e}")
            self.compute_clap = False

    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        log_scale: bool = True,
    ) -> np.ndarray:
        """Extract mel spectrogram features from audio."""

        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window="hann",
        )

        # Convert to magnitude spectrogram
        magnitude = np.abs(stft)

        # Apply mel filterbank
        mel_spec = np.dot(self.mel_basis, magnitude)

        # Convert to log scale
        if log_scale:
            mel_spec = np.log(mel_spec + 1e-8)

        return mel_spec

    def extract_audio_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features."""

        features = {}

        # Spectral features
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)

        # Spectral centroid
        features["spectral_centroid"] = librosa.feature.spectral_centroid(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )

        # Spectral rolloff
        features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )

        # Spectral bandwidth
        features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )

        # Zero crossing rate
        features["zcr"] = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)

        # MFCCs
        features["mfcc"] = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length
        )

        # Chroma features
        features["chroma"] = librosa.feature.chroma_stft(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )

        # Mel spectrogram
        features["mel_spectrogram"] = self.extract_mel_spectrogram(audio)

        # Tonnetz (harmonic network)
        features["tonnetz"] = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)

        return features

    def compute_spectral_distance(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        metric: str = "l2",
    ) -> float:
        """Compute spectral distance between two audio signals."""

        # Extract mel spectrograms
        mel1 = self.extract_mel_spectrogram(audio1)
        mel2 = self.extract_mel_spectrogram(audio2)

        # Ensure same temporal dimension
        min_frames = min(mel1.shape[1], mel2.shape[1])
        mel1 = mel1[:, :min_frames]
        mel2 = mel2[:, :min_frames]

        # Compute distance
        if metric == "l2":
            distance = np.linalg.norm(mel1 - mel2)
        elif metric == "l1":
            distance = np.sum(np.abs(mel1 - mel2))
        elif metric == "cosine":
            mel1_flat = mel1.flatten()
            mel2_flat = mel2.flatten()
            distance = 1 - cosine_similarity([mel1_flat], [mel2_flat])[0, 0]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return float(distance)

    def compute_frechet_audio_distance(
        self,
        generated_audio: List[np.ndarray],
        reference_audio: List[np.ndarray],
    ) -> float:
        """
        Compute Fréchet Audio Distance (FAD) between generated and reference audio.

        FAD is computed using VGGish embeddings and measures the distance
        between the distributions of generated and reference audio.
        """

        if not self.compute_fad:
            logger.warning("FAD computation disabled")
            return float("inf")

        try:
            # Extract VGGish embeddings
            gen_embeddings = self._extract_vggish_embeddings(generated_audio)
            ref_embeddings = self._extract_vggish_embeddings(reference_audio)

            # Compute statistics
            mu_gen = np.mean(gen_embeddings, axis=0)
            sigma_gen = np.cov(gen_embeddings, rowvar=False)

            mu_ref = np.mean(ref_embeddings, axis=0)
            sigma_ref = np.cov(ref_embeddings, rowvar=False)

            # Compute FAD
            fad = self._compute_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)

            return float(fad)

        except Exception as e:
            logger.error(f"Failed to compute FAD: {e}")
            return float("inf")

    def _extract_vggish_embeddings(self, audio_list: List[np.ndarray]) -> np.ndarray:
        """Extract VGGish embeddings from audio list."""

        embeddings = []

        for audio in audio_list:
            # Ensure audio is in the right format for VGGish
            if len(audio.shape) == 1:
                audio = audio[np.newaxis, :]  # Add channel dimension

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()

            # Extract embeddings
            with torch.no_grad():
                embedding = self.vggish_model(audio_tensor)
                embeddings.append(embedding.numpy())

        return np.vstack(embeddings)

    def _compute_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
    ) -> float:
        """Compute Fréchet distance between two multivariate Gaussians."""

        # Compute mean difference
        diff = mu1 - mu2

        # Compute matrix square root
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # Handle numerical instability
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Compute FAD
        fad = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

        return fad

    def compute_inception_score(
        self,
        audio_list: List[np.ndarray],
        splits: int = 10,
    ) -> Tuple[float, float]:
        """
        Compute Inception Score for audio quality assessment.

        This uses a pre-trained audio classifier to assess the quality
        and diversity of generated audio.
        """

        if not self.compute_inception_score:
            return 0.0, 0.0

        try:
            # For simplicity, we'll use spectral diversity as a proxy
            # In practice, you'd use a pre-trained audio classifier

            scores = []

            for i in range(splits):
                # Sample subset
                subset_size = len(audio_list) // splits
                start_idx = i * subset_size
                end_idx = start_idx + subset_size
                subset = audio_list[start_idx:end_idx]

                if not subset:
                    continue

                # Extract features
                features_list = []
                for audio in subset:
                    features = self.extract_audio_features(audio)
                    # Use MFCC features for diversity assessment
                    mfcc_mean = np.mean(features["mfcc"], axis=1)
                    features_list.append(mfcc_mean)

                if not features_list:
                    continue

                features_array = np.array(features_list)

                # Compute "probabilities" using softmax of feature similarities
                similarities = cosine_similarity(features_array)
                probabilities = np.exp(similarities) / np.sum(
                    np.exp(similarities), axis=1, keepdims=True
                )

                # Compute KL divergence (proxy for inception score)
                marginal = np.mean(probabilities, axis=0)
                kl_divs = []
                for p in probabilities:
                    kl_div = np.sum(p * np.log(p / (marginal + 1e-8) + 1e-8))
                    kl_divs.append(kl_div)

                score = np.exp(np.mean(kl_divs))
                scores.append(score)

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            return float(mean_score), float(std_score)

        except Exception as e:
            logger.error(f"Failed to compute Inception Score: {e}")
            return 0.0, 0.0

    def compute_signal_to_noise_ratio(self, audio: np.ndarray) -> float:
        """Compute signal-to-noise ratio of audio."""

        # Simple SNR estimation using energy-based voice activity detection
        # Split audio into frames
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)  # 10ms hop

        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)

        # Compute frame energies
        energies = np.sum(frames**2, axis=0)

        # Estimate noise floor (bottom 20% of energies)
        noise_floor = np.percentile(energies, 20)

        # Estimate signal level (top 20% of energies)
        signal_level = np.percentile(energies, 80)

        # Compute SNR in dB
        snr_db = 10 * np.log10(signal_level / (noise_floor + 1e-8))

        return float(snr_db)

    def compute_harmonic_percussive_ratio(self, audio: np.ndarray) -> float:
        """Compute ratio of harmonic to percussive components."""

        # Decompose into harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio)

        # Compute energy ratio
        harmonic_energy = np.sum(harmonic**2)
        percussive_energy = np.sum(percussive**2)

        ratio = harmonic_energy / (percussive_energy + 1e-8)

        return float(ratio)

    def compute_tempo_stability(self, audio: np.ndarray) -> float:
        """Assess tempo stability of the audio."""

        try:
            # Extract tempo over time
            onset_frames = librosa.onset.onset_detect(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )

            if len(onset_frames) < 4:
                return 0.0  # Not enough onsets to assess tempo

            # Convert to time
            onset_times = librosa.frames_to_time(
                onset_frames, sr=self.sample_rate, hop_length=self.hop_length
            )

            # Compute inter-onset intervals
            intervals = np.diff(onset_times)

            if len(intervals) == 0:
                return 0.0

            # Compute tempo stability as inverse of coefficient of variation
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)

            if mean_interval == 0:
                return 0.0

            cv = std_interval / mean_interval
            stability = 1.0 / (1.0 + cv)

            return float(stability)

        except Exception as e:
            logger.warning(f"Failed to compute tempo stability: {e}")
            return 0.0

    def compute_pitch_stability(self, audio: np.ndarray) -> float:
        """Assess pitch stability of the audio."""

        try:
            # Extract fundamental frequency
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=self.sample_rate,
            )

            # Filter out unvoiced segments
            voiced_f0 = f0[voiced_flag]

            if len(voiced_f0) < 10:
                return 0.0

            # Compute pitch stability
            mean_f0 = np.mean(voiced_f0)
            std_f0 = np.std(voiced_f0)

            if mean_f0 == 0:
                return 0.0

            cv = std_f0 / mean_f0
            stability = 1.0 / (1.0 + cv)

            return float(stability)

        except Exception as e:
            logger.warning(f"Failed to compute pitch stability: {e}")
            return 0.0

    def evaluate_audio_quality(
        self,
        generated_audio: Union[np.ndarray, List[np.ndarray]],
        reference_audio: Optional[List[np.ndarray]] = None,
        texts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Comprehensive audio quality evaluation.

        Args:
            generated_audio: Generated audio samples
            reference_audio: Reference audio for comparison
            texts: Text prompts for text-audio alignment evaluation

        Returns:
            Dictionary of quality metrics
        """

        # Ensure generated_audio is a list
        if isinstance(generated_audio, np.ndarray):
            if generated_audio.ndim == 1:
                generated_audio = [generated_audio]
            else:
                generated_audio = [generated_audio[i] for i in range(generated_audio.shape[0])]

        metrics = {}

        # Basic audio quality metrics
        snr_scores = []
        hp_ratios = []
        tempo_stabilities = []
        pitch_stabilities = []

        for audio in generated_audio:
            # Ensure audio is 1D
            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            # Basic quality metrics
            snr = self.compute_signal_to_noise_ratio(audio)
            hp_ratio = self.compute_harmonic_percussive_ratio(audio)
            tempo_stability = self.compute_tempo_stability(audio)
            pitch_stability = self.compute_pitch_stability(audio)

            snr_scores.append(snr)
            hp_ratios.append(hp_ratio)
            tempo_stabilities.append(tempo_stability)
            pitch_stabilities.append(pitch_stability)

        # Aggregate basic metrics
        metrics["snr_mean"] = float(np.mean(snr_scores))
        metrics["snr_std"] = float(np.std(snr_scores))
        metrics["harmonic_percussive_ratio"] = float(np.mean(hp_ratios))
        metrics["tempo_stability"] = float(np.mean(tempo_stabilities))
        metrics["pitch_stability"] = float(np.mean(pitch_stabilities))

        # Fréchet Audio Distance
        if reference_audio is not None and self.compute_fad:
            fad_score = self.compute_frechet_audio_distance(generated_audio, reference_audio)
            metrics["fad"] = fad_score

        # Inception Score
        if self.compute_inception_score:
            is_mean, is_std = self.compute_inception_score(generated_audio)
            metrics["inception_score_mean"] = is_mean
            metrics["inception_score_std"] = is_std

        # Diversity metrics
        if len(generated_audio) > 1:
            diversity_score = self._compute_diversity(generated_audio)
            metrics["diversity"] = diversity_score

        return metrics

    def _compute_diversity(self, audio_list: List[np.ndarray]) -> float:
        """Compute diversity score among generated audio samples."""

        # Extract features for all samples
        features_list = []
        for audio in audio_list:
            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            mel_spec = self.extract_mel_spectrogram(audio)
            features_list.append(mel_spec.flatten())

        # Compute pairwise similarities
        similarities = []
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                sim = cosine_similarity([features_list[i]], [features_list[j]])[0, 0]
                similarities.append(sim)

        # Diversity is 1 - average similarity
        if similarities:
            diversity = 1.0 - np.mean(similarities)
        else:
            diversity = 0.0

        return float(diversity)

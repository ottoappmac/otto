"""Voice Activity Detection.

:class:`SileroVAD` accumulates 16 kHz int16 PCM chunks from the microphone
and returns a complete utterance when a configurable silence period elapses
after speech ends.

The implementation uses an energy (RMS) heuristic with an adaptive noise
floor — no external ONNX model required.  Call :meth:`process_chunk` with
each raw ``bytes`` chunk; it returns the accumulated utterance bytes when
the silence threshold is reached, or ``None`` otherwise.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_CHUNK_SAMPLES = 512  # ~32 ms per chunk at 16 kHz


class SileroVAD:
    """Energy-based VAD that buffers speech until ``silence_secs`` of quiet.

    Despite the name (kept for API compatibility), this implementation
    does not use the Silero ONNX model.  A simple RMS energy threshold
    with adaptive noise-floor calibration is used instead.
    """

    def __init__(self, *, silence_secs: float = 1.0) -> None:
        self._silence_frames_target = max(1, int(silence_secs * _SAMPLE_RATE / _CHUNK_SAMPLES))
        self._buffer: list[bytes] = []
        self._in_speech = False
        self._silence_frames = 0
        # Noise-floor calibrated from the first ~1 s of background audio.
        self._noise_floor: float = 0.002
        self._calibration_frames: int = 0

    def process_chunk(self, chunk: bytes) -> bytes | None:
        """Process one mic chunk.  Returns utterance bytes when complete, else None."""
        import numpy as np

        if not chunk:
            return None

        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32_768.0
        energy = float(np.sqrt(np.mean(audio ** 2)))  # RMS

        if self._calibration_frames < 30:
            self._noise_floor = max(self._noise_floor, energy * 0.5)
            self._calibration_frames += 1

        threshold = max(self._noise_floor * 3.0, 0.005)
        is_speech = energy > threshold

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                logger.debug("VAD: speech start (rms=%.4f threshold=%.4f)", energy, threshold)
            self._silence_frames = 0
            self._buffer.append(chunk)

        elif self._in_speech:
            self._buffer.append(chunk)
            self._silence_frames += 1
            if self._silence_frames >= self._silence_frames_target:
                utterance = b"".join(self._buffer)
                logger.debug("VAD: utterance done, %d bytes", len(utterance))
                self._buffer = []
                self._in_speech = False
                self._silence_frames = 0
                return utterance

        return None

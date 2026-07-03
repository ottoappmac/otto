"""Wake-word detector backed by openWakeWord (ONNX).

:class:`WakeWordDetector` runs inference on every ~32 ms mic chunk and fires
``on_wake()`` when the model's score crosses ``threshold``.  The callback is
invoked from the VAD thread, so it must be thread-safe (typically a
``loop.call_soon_threadsafe`` wrapper).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent / "models"


class WakeWordDetector:
    """openWakeWord-backed detector.

    Parameters
    ----------
    model_name:
        ``"hey_otto"`` → bundled ONNX at ``backend/voice/models/hey_otto.onnx``.
        Any other value is passed directly to the OWW loader (HF repo ID or
        absolute path to a ``.onnx`` file).
    threshold:
        Activation score in [0, 1].  Higher = fewer false positives.
    on_wake:
        Called (no arguments) when the wake word fires.  Must be thread-safe.
    """

    def __init__(
        self,
        *,
        model_name: str = "hey_otto",
        threshold: float = 0.5,
        on_wake: Optional[Callable[[], None]] = None,
    ) -> None:
        self._threshold = threshold
        self._on_wake = on_wake
        self._model = None
        self._cooldown_frames = 0  # debounce: skip N frames after firing (~1.6 s)

        model_path = self._resolve_model(model_name)
        self._load(model_path)

    @staticmethod
    def _resolve_model(name: str) -> str:
        if name == "hey_otto":
            bundled = _MODELS_DIR / "hey_otto.onnx"
            if bundled.is_file():
                return str(bundled)
            logger.warning("Bundled hey_otto.onnx not found at %s; falling back to OWW lookup", bundled)
            return name
        p = Path(name)
        if p.is_file():
            return str(p)
        return name

    def _load(self, model_path: str) -> None:
        try:
            from openwakeword.model import Model  # type: ignore[import]
            self._model = Model(wakeword_models=[model_path], inference_framework="onnx")
            logger.info("Wake-word model loaded: %s", model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load wake-word model '%s': %s", model_path, exc)
            self._model = None

    def process_chunk(self, chunk: bytes) -> None:
        """Run inference on one ~32 ms mic chunk (16 kHz int16 PCM)."""
        if self._model is None:
            return

        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1
            return

        import numpy as np

        audio = np.frombuffer(chunk, dtype=np.int16)
        try:
            prediction = self._model.predict(audio)
        except Exception as exc:  # noqa: BLE001
            logger.debug("wake-word predict error: %s", exc)
            return

        best = max(prediction.values(), default=0.0)
        if best >= self._threshold:
            logger.info("Wake word detected (score=%.3f)", best)
            self._cooldown_frames = 50  # ~1.6 s debounce at 32 ms/chunk
            if self._on_wake:
                self._on_wake()

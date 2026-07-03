"""Microphone capture and speaker playback via sounddevice (PortAudio).

:class:`AudioInput`  — 16 kHz mono mic capture (for STT / VAD / wake word).
:class:`AudioOutput` — speaker playback queue (kept for potential future use).

All imports of ``sounddevice`` and ``numpy`` are deferred so the module
can be imported on machines where PortAudio is not installed.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class AudioInput:
    """Microphone capture that delivers 16 kHz int16 PCM chunks via callback.

    Each chunk is ``BLOCKSIZE`` samples (~32 ms).  The callback is invoked
    from the PortAudio thread and must be non-blocking.
    """

    SAMPLE_RATE = 16_000
    BLOCKSIZE = 512  # ~32 ms at 16 kHz

    def __init__(
        self,
        *,
        device: "str | int | None" = None,
        callback: "Callable[[bytes], None]",
    ) -> None:
        self._device = device
        self._callback = callback
        self._stream = None

    def start(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd

            def _cb(indata: "np.ndarray", frames: int, _time, status) -> None:  # noqa: ANN001
                if status:
                    logger.debug("AudioInput status: %s", status)
                chunk = (indata[:, 0] * 32_767).astype(np.int16).tobytes()
                self._callback(chunk)

            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=self._device or None,
                blocksize=self.BLOCKSIZE,
                callback=_cb,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not open microphone: {exc}") from exc

    def stop(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None


def list_input_devices() -> list[dict]:
    """Return available input devices as ``[{index, name, channels, default}]``."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        return [
            {
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "default": i == default_in,
            }
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("list_input_devices failed: %s", exc)
        return []


class AudioOutput:
    """Thread-safe speaker playback queue.

    Enqueue float32 numpy arrays (any sample rate; specify ``sample_rate``
    matching the source).  Plays sequentially via a persistent OutputStream
    callback — no per-chunk stream teardown so there are no gaps between
    synthesised sentences.  Calling :meth:`cancel` drains the queue and
    clears the in-flight buffer for barge-in behaviour.
    """

    def __init__(self, *, device: str | int | None = None, sample_rate: int = 24_000) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="audio-output")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.cancel()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def enqueue(self, pcm_array) -> None:
        """Enqueue a float32 numpy array for playback."""
        self._q.put(pcm_array)

    def cancel(self) -> None:
        """Drain queue and interrupt current playback immediately (barge-in)."""
        self._cancel_event.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _worker(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            logger.warning("sounddevice not available — AudioOutput worker exiting")
            return

        # Per-callback audio buffer (only ever touched by the PortAudio thread).
        _buf: list = [np.zeros(0, dtype=np.float32)]

        def _callback(outdata: "np.ndarray", frames: int, _time, _status) -> None:  # noqa: ANN001
            # Barge-in / cancel: drain local buffer and output silence.
            if self._cancel_event.is_set():
                _buf[0] = np.zeros(0, dtype=np.float32)
                self._cancel_event.clear()
                outdata.fill(0)
                return

            # Top up the local buffer from the synthesis queue.
            while len(_buf[0]) < frames:
                try:
                    chunk = self._q.get_nowait()
                    _buf[0] = np.concatenate((_buf[0], chunk))
                except queue.Empty:
                    break

            available = len(_buf[0])
            if available >= frames:
                outdata[:, 0] = _buf[0][:frames]
                _buf[0] = _buf[0][frames:]
            elif available > 0:
                outdata[:available, 0] = _buf[0]
                outdata[available:, 0] = 0.0
                _buf[0] = np.zeros(0, dtype=np.float32)
            else:
                outdata.fill(0)

        try:
            with sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                device=self._device or None,
                callback=_callback,
                blocksize=2048,
            ):
                while self._running:
                    import time
                    time.sleep(0.05)
        except Exception as exc:  # noqa: BLE001
            logger.debug("AudioOutput stream error: %s", exc)


def list_output_devices() -> list[dict]:
    """Return available output devices as ``[{index, name, channels, default}]``."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_out = sd.default.device[1]
        return [
            {
                "index": i,
                "name": d["name"],
                "channels": d["max_output_channels"],
                "default": i == default_out,
            }
            for i, d in enumerate(devices)
            if d["max_output_channels"] > 0
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("list_output_devices failed: %s", exc)
        return []

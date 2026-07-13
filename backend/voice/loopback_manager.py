"""System-audio (loopback) + microphone transcription manager.

Captures audio from one or both sources and transcribes it in real time with
the existing on-device Whisper pipeline:

  * ``system`` — what the Mac is playing (speakers/headphones) via the native
    ``otto-audiotap`` Core Audio process-tap helper (macOS 14.4+, no driver,
    no rerouting).
  * ``mic``    — the microphone via ``sounddevice`` (PortAudio).

Both sources can run at once (e.g. to capture both sides of a call). Each has
its own energy VAD, and events are tagged with their ``source`` so the UI can
distinguish "System" from "Mic".

This is intentionally separate from :class:`VoiceManager` (the push-to-talk /
wake-word microphone pipeline) so the two features don't interfere.

States
------
idle       — not capturing.
recording  — at least one source active.

Events emitted to the frontend via /ws/transcribe
-------------------------------------------------
{"type": "state",   "state": "idle|recording"}
{"type": "partial", "text": "...", "source": "system|mic"}
{"type": "segment", "text": "...", "ts": <epoch>, "source": "system|mic"}
{"type": "level",   "rms": <0..1>, "source": "system|mic"}
{"type": "error",   "message": "..."}
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_BYTES_PER_SAMPLE = 2
_CHUNK_SAMPLES = 512  # ~32 ms at 16 kHz
_CHUNK_BYTES = _CHUNK_SAMPLES * _BYTES_PER_SAMPLE
_LEVEL_EMIT_INTERVAL = 0.1  # seconds between level meter updates

SOURCE_SYSTEM = "system"
SOURCE_MIC = "mic"
_VALID_SOURCES = (SOURCE_SYSTEM, SOURCE_MIC)


class LoopbackState(str, Enum):
    idle = "idle"
    recording = "recording"


def _resolve_helper_path() -> Optional[str]:
    """Locate the otto-audiotap helper binary (env override, bundle, dev build)."""
    env = os.environ.get("OTTO_AUDIOTAP_BIN")
    if env and Path(env).is_file():
        return env

    candidates: list[Path] = []

    # Frozen PyInstaller build: helper is bundled next to the backend exe under
    # ../audiotap/otto-audiotap (see tauri.conf.json resources mapping).
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir.parent / "audiotap" / "otto-audiotap")
        candidates.append(exe_dir / "otto-audiotap")

    # Dev build produced by app/src-tauri/build-audiotap.sh.
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(
        repo_root / "app" / "src-tauri" / "audiotap" / ".build" / "release" / "otto-audiotap"
    )

    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def helper_available() -> bool:
    return _resolve_helper_path() is not None


def mic_available() -> bool:
    """True if sounddevice can enumerate at least one input device."""
    try:
        from backend.voice.audio_io import list_input_devices

        return len(list_input_devices()) > 0
    except Exception:  # noqa: BLE001
        return False


def macos_supported() -> bool:
    """True on macOS 14.4+ where Core Audio process taps exist."""
    if sys.platform != "darwin":
        return False
    try:
        import platform

        parts = platform.mac_ver()[0].split(".")
        major = int(parts[0]) if parts and parts[0] else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    except Exception:  # noqa: BLE001
        return False
    return (major, minor) >= (14, 4)


class _SourceState:
    """Per-source energy-VAD + segmentation state."""

    def __init__(self) -> None:
        self.buffer: list[bytes] = []
        self.in_speech = False
        self.silence_frames = 0
        self.noise_floor = 0.005
        self.partial_inflight = False
        self.last_partial_ts = 0.0
        self.last_level_ts = 0.0


class LoopbackManager:
    """Singleton owning the system-audio + mic capture/transcription pipeline."""

    def __init__(self) -> None:
        self._state = LoopbackState.idle
        self._lock = asyncio.Lock()
        self._broadcast_queues: list[asyncio.Queue] = []

        # Stable, run-scoped id stamped on every emitted event.  The SAME event
        # object is delivered to every connected client queue, so all copies
        # share the id — letting the frontend collapse duplicates when more than
        # one socket happens to be open.  The run_id (regenerated per process)
        # keeps ids unique across backend restarts so seqs can't collide.
        self._event_seq = 0
        self._run_id = uuid.uuid4().hex[:8]

        self._cfg: dict[str, Any] = {}

        self._active: set[str] = set()
        self._src: dict[str, _SourceState] = {}

        # Speech-model readiness. Capture is gated on the Whisper model being
        # fully downloaded so the first run doesn't silently hang while ~1.5 GB
        # streams from HuggingFace. Progress is surfaced as "model" events.
        self._model_ready = False
        self._model_id_checked: Optional[str] = None
        self._model_preparing = False
        self._model_progress = 0.0
        self._model_task: Optional[asyncio.Task] = None

        # System-audio helper.
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._sys_read_task: Optional[asyncio.Task] = None
        self._sys_err_task: Optional[asyncio.Task] = None

        # Microphone.
        self._mic_input = None
        self._mic_queue: Optional[asyncio.Queue] = None
        self._mic_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Config + broadcast
    # ------------------------------------------------------------------

    def configure(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        from backend.voice import stt as _stt

        model = cfg.get("stt_model", "mlx-community/whisper-large-v3-turbo")
        _stt.configure(model)
        # If the model was switched, forget the previous readiness verdict so
        # the new model is re-checked (and downloaded) before capture.
        if model != self._model_id_checked:
            self._model_ready = False
            self._model_id_checked = None

    def add_client(self, q: asyncio.Queue) -> None:
        self._broadcast_queues.append(q)

    def remove_client(self, q: asyncio.Queue) -> None:
        try:
            self._broadcast_queues.remove(q)
        except ValueError:
            pass

    def _emit(self, event: dict[str, Any]) -> None:
        self._event_seq += 1
        event["eid"] = f"{self._run_id}:{self._event_seq}"
        for q in list(self._broadcast_queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def _set_state(self, state: LoopbackState) -> None:
        if self._state != state:
            self._state = state
            self._emit({"type": "state", "state": state.value})

    @property
    def state(self) -> LoopbackState:
        return self._state

    @property
    def active_sources(self) -> set[str]:
        return set(self._active)

    # ------------------------------------------------------------------
    # Speech-model readiness / download
    # ------------------------------------------------------------------

    def _current_model_id(self) -> str:
        return self._cfg.get("stt_model") or "mlx-community/whisper-large-v3-turbo"

    def _emit_model(self, status: str, **extra: Any) -> None:
        """Emit a model-status event: status is ready|downloading|error."""
        event: dict[str, Any] = {"type": "model", "status": status, "model": self._current_model_id()}
        event.update(extra)
        self._emit(event)

    @property
    def model_ready(self) -> bool:
        return self._model_ready

    async def ensure_model_ready(self, *, notify: bool = True) -> bool:
        """Ensure the speech model is downloaded, kicking off a download if not.

        Returns True if the model is already present (capture can start), or
        False if a download was started / is in progress (capture must wait).
        Progress is broadcast as ``{"type":"model", ...}`` events either way.
        """
        from backend.voice import stt as _stt

        model = self._current_model_id()

        if self._model_ready and self._model_id_checked == model:
            if notify:
                self._emit_model("ready")
            return True

        # Cheap cache probe off the event loop.
        if await asyncio.to_thread(_stt.is_model_ready, model):
            self._model_ready = True
            self._model_id_checked = model
            if notify:
                self._emit_model("ready")
            return True

        # Not cached — start a background download (or report the running one).
        if not self._model_preparing:
            self._model_preparing = True
            self._model_progress = 0.0
            self._model_task = asyncio.create_task(
                self._download_model(model), name="loopback_model_dl"
            )
        elif notify:
            self._emit_model("downloading", progress=round(self._model_progress, 3))
        return False

    async def _download_model(self, model: str) -> None:
        from backend.voice import stt as _stt

        self._emit_model("downloading", progress=0.0)
        total = await asyncio.to_thread(_stt.expected_total_bytes, model)

        stop = asyncio.Event()

        async def _poll() -> None:
            while not stop.is_set():
                done = await asyncio.to_thread(_stt.cached_bytes, model)
                if total:
                    # Monotonic and capped below 1.0 until the download actually
                    # finishes, so the bar never shows 100% prematurely.
                    self._model_progress = max(self._model_progress, min(0.999, done / total))
                    self._emit_model("downloading", progress=round(self._model_progress, 3))
                else:
                    self._emit_model("downloading")  # indeterminate
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.7)
                except asyncio.TimeoutError:
                    pass

        poll_task = asyncio.create_task(_poll(), name="loopback_model_poll")
        try:
            await asyncio.to_thread(_stt.download_model, model)
            self._model_ready = True
            self._model_id_checked = model
            self._model_progress = 1.0
            self._emit_model("ready")
            logger.info("Speech model ready: %s", model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Speech model download failed (%s): %s", model, exc)
            self._emit_model(
                "error",
                message=f"Couldn't download the speech model ({model}): {exc}",
            )
        finally:
            stop.set()
            poll_task.cancel()
            self._model_preparing = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, sources: Optional[list[str]] = None) -> None:
        async with self._lock:
            if self._state != LoopbackState.idle:
                return

            # Gate capture on the speech model being present. If it isn't,
            # ensure_model_ready kicks off the download (emitting "model"
            # progress events) and we bail out — the UI keeps Record disabled
            # until it receives {"type":"model","status":"ready"}.
            if not await self.ensure_model_ready():
                return

            requested = [s for s in (sources or [SOURCE_SYSTEM]) if s in _VALID_SOURCES]
            if not requested:
                requested = [SOURCE_SYSTEM]

            started: set[str] = set()

            # Mic (PortAudio/CoreAudio HAL InputStream) must be opened *before*
            # the system-audio helper creates its Core Audio process tap +
            # aggregate device. Opening the mic stream while an aggregate
            # device is being registered by another process can deadlock
            # PortAudio's device-list scan inside Pa_OpenStream/Pa_StartStream
            # (observed hang inside the CoreAudio HAL's global device mutex).
            # Starting mic first sidesteps that race entirely.
            if SOURCE_MIC in requested:
                if self._start_mic():
                    started.add(SOURCE_MIC)

            if SOURCE_SYSTEM in requested:
                if await self._start_system():
                    started.add(SOURCE_SYSTEM)

            if not started:
                # Errors already emitted by the start helpers.
                return

            # Warm the Whisper model so the first segment isn't multi-second.
            asyncio.create_task(self._warmup())

            self._active = started
            self._set_state(LoopbackState.recording)
            logger.info("Loopback capture started (sources=%s)", ",".join(sorted(started)))

    async def _start_system(self) -> bool:
        if not macos_supported():
            self._emit({"type": "error", "message": "System audio capture requires macOS 14.4 or later."})
            return False
        helper = _resolve_helper_path()
        if not helper:
            self._emit({
                "type": "error",
                "message": "System audio helper not found. Rebuild with app/src-tauri/build-audiotap.sh.",
            })
            return False

        try:
            self._proc = await asyncio.create_subprocess_exec(
                helper,
                "--sample-rate",
                str(_SAMPLE_RATE),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:  # noqa: BLE001
            self._emit({"type": "error", "message": f"Could not start system audio capture: {exc}"})
            return False

        self._src[SOURCE_SYSTEM] = _SourceState()
        self._sys_read_task = asyncio.create_task(self._system_read_loop(), name="loopback_sys_read")
        self._sys_err_task = asyncio.create_task(self._system_err_loop(), name="loopback_sys_err")
        return True

    def _start_mic(self) -> bool:
        try:
            from backend.voice.audio_io import AudioInput
        except Exception as exc:  # noqa: BLE001
            self._emit({"type": "error", "message": f"Microphone unavailable: {exc}"})
            return False

        loop = asyncio.get_event_loop()
        self._mic_queue = asyncio.Queue(maxsize=512)

        def _mic_callback(chunk: bytes) -> None:
            try:
                loop.call_soon_threadsafe(self._mic_queue.put_nowait, chunk)
            except Exception:  # noqa: BLE001
                pass

        device = self._cfg.get("mic_device") or None
        self._mic_input = AudioInput(device=device, callback=_mic_callback)
        try:
            self._mic_input.start()
        except Exception as exc:  # noqa: BLE001
            self._emit({"type": "error", "message": f"Could not open microphone: {exc}"})
            self._mic_input = None
            self._mic_queue = None
            return False

        self._src[SOURCE_MIC] = _SourceState()
        self._mic_task = asyncio.create_task(self._mic_read_loop(), name="loopback_mic_read")
        return True

    async def stop(self) -> None:
        async with self._lock:
            if self._state == LoopbackState.idle:
                return

            # Flush any in-progress speech per source as a final segment.
            for source, st in self._src.items():
                if st.buffer:
                    pending = b"".join(st.buffer)
                    st.buffer = []
                    if pending:
                        asyncio.create_task(self._finalize(source, pending))

            # System audio teardown.
            proc = self._proc
            self._proc = None
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
            for task in (self._sys_read_task, self._sys_err_task):
                if task and not task.done():
                    task.cancel()
            self._sys_read_task = None
            self._sys_err_task = None

            # Microphone teardown.
            if self._mic_queue is not None:
                try:
                    self._mic_queue.put_nowait(None)
                except Exception:  # noqa: BLE001
                    pass
            if self._mic_task and not self._mic_task.done():
                self._mic_task.cancel()
            self._mic_task = None
            if self._mic_input is not None:
                self._mic_input.stop()
                self._mic_input = None
            self._mic_queue = None

            self._src = {}
            self._active = set()
            self._set_state(LoopbackState.idle)
            logger.info("Loopback capture stopped")

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    async def _warmup(self) -> None:
        try:
            from backend.voice import stt as _stt

            silence = b"\x00\x00" * int(_SAMPLE_RATE * 0.1)
            await _stt.transcribe(silence, language=self._cfg.get("stt_language") or None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Loopback warmup skipped: %s", exc)

    # ------------------------------------------------------------------
    # Producers
    # ------------------------------------------------------------------

    async def _system_err_loop(self) -> None:
        proc = self._proc
        if not proc or not proc.stderr:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                logger.debug("audiotap: %s", text)
                if "FATAL" in text or "System Audio Recording" in text:
                    self._emit({"type": "error", "message": text.replace("[audiotap] ", "")})
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            return

    async def _system_read_loop(self) -> None:
        proc = self._proc
        if not proc or not proc.stdout:
            return
        try:
            while True:
                try:
                    chunk = await proc.stdout.readexactly(_CHUNK_BYTES)
                except asyncio.IncompleteReadError as exc:
                    chunk = exc.partial
                    if not chunk:
                        break
                if not chunk:
                    break
                await self._handle_chunk(SOURCE_SYSTEM, chunk)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Loopback system read loop error: %s", exc)
        finally:
            if self._state == LoopbackState.recording and self._proc is proc:
                self._emit({"type": "error", "message": "System audio capture ended unexpectedly."})
                asyncio.create_task(self.stop())

    async def _mic_read_loop(self) -> None:
        q = self._mic_queue
        if q is None:
            return
        try:
            while True:
                chunk = await q.get()
                if chunk is None:
                    return
                await self._handle_chunk(SOURCE_MIC, chunk)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Loopback mic read loop error: %s", exc)

    # ------------------------------------------------------------------
    # Segmentation (per source)
    # ------------------------------------------------------------------

    async def _handle_chunk(self, source: str, chunk: bytes) -> None:
        import numpy as np

        st = self._src.get(source)
        if st is None:
            return

        silence_secs = float(self._cfg.get("loopback_vad_silence_secs", 0.7))
        silence_target = max(1, int(silence_secs * _SAMPLE_RATE / _CHUNK_SAMPLES))
        max_secs = float(self._cfg.get("loopback_max_segment_secs", 12.0))
        max_bytes = int(max_secs * _SAMPLE_RATE * _BYTES_PER_SAMPLE)
        live_partials = bool(self._cfg.get("loopback_live_partials", True))
        partial_interval = float(self._cfg.get("loopback_partial_interval_secs", 1.5))

        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32_768.0
        energy = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0

        now = time.monotonic()
        if now - st.last_level_ts >= _LEVEL_EMIT_INTERVAL:
            st.last_level_ts = now
            self._emit({"type": "level", "rms": min(1.0, energy * 4.0), "source": source})

        # Adaptive noise floor: only track it while NOT in speech so loud audio
        # can't poison the baseline.
        threshold = max(st.noise_floor * 2.5, 0.006)
        is_speech = energy > threshold
        if not is_speech:
            st.noise_floor = 0.95 * st.noise_floor + 0.05 * energy

        if is_speech:
            st.in_speech = True
            st.silence_frames = 0
            st.buffer.append(chunk)
        elif st.in_speech:
            st.buffer.append(chunk)
            st.silence_frames += 1
            if st.silence_frames >= silence_target:
                utterance = b"".join(st.buffer)
                st.buffer = []
                st.in_speech = False
                st.silence_frames = 0
                st.last_partial_ts = 0.0
                if utterance:
                    asyncio.create_task(self._finalize(source, utterance))
                return

        buffered = sum(len(b) for b in st.buffer)
        if st.in_speech and buffered >= max_bytes:
            utterance = b"".join(st.buffer)
            st.buffer = []
            st.silence_frames = 0
            st.last_partial_ts = 0.0
            if utterance:
                asyncio.create_task(self._finalize(source, utterance))
            return

        if (
            live_partials
            and st.in_speech
            and not st.partial_inflight
            and buffered > 0
            and (now - st.last_partial_ts) >= partial_interval
        ):
            st.last_partial_ts = now
            st.partial_inflight = True
            asyncio.create_task(self._partial(source, b"".join(st.buffer)))

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def _partial(self, source: str, pcm_bytes: bytes) -> None:
        try:
            from backend.voice import stt as _stt

            text = await _stt.transcribe(pcm_bytes, language=self._cfg.get("stt_language") or None)
            if text.strip():
                self._emit({"type": "partial", "text": text, "source": source})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Loopback partial error: %s", exc)
        finally:
            st = self._src.get(source)
            if st is not None:
                st.partial_inflight = False

    async def _finalize(self, source: str, pcm_bytes: bytes) -> None:
        try:
            from backend.voice import stt as _stt

            text = await _stt.transcribe(pcm_bytes, language=self._cfg.get("stt_language") or None)
            if text.strip():
                self._emit({"type": "segment", "text": text, "ts": time.time(), "source": source})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Loopback STT error: %s", exc)
            self._emit({"type": "error", "message": str(exc)})


_manager = LoopbackManager()


def get_loopback_manager() -> LoopbackManager:
    return _manager

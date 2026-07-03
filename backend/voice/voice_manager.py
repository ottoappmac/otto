"""Central voice-session state machine (STT + wake word; no TTS).

States
------
idle          — voice disabled or not started; no mic capture.
listening     — wake-word mode: mic open, waiting for trigger phrase.
capturing     — speech detected (PTT held OR wake word fired); buffering audio.
transcribing  — utterance complete; running Whisper STT.

Transitions
-----------
idle → listening      start() called, mode = wakeword
idle → capturing      ptt_start() called, mode = ptt
listening → capturing wake word detected  OR  ptt_start()
capturing → transcribing  end of utterance (VAD) OR ptt_stop()
transcribing → idle / listening   transcript emitted via queue

Events emitted to the frontend via /ws/voice
--------------------------------------------
{"type": "state",       "state": <state>}
{"type": "partial",     "text": <partial transcript>}
{"type": "transcript",  "text": <final transcript>}
{"type": "wake"}
{"type": "error",       "message": <msg>}
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class VoiceState(str, Enum):
    idle = "idle"
    listening = "listening"
    capturing = "capturing"
    transcribing = "transcribing"


class VoiceManager:
    """Singleton that owns the audio pipeline for one Otto voice session.

    Instantiated once at server startup; configured per-user settings change.
    Multiple frontend WebSocket clients share the single manager state.
    """

    def __init__(self) -> None:
        self._state = VoiceState.idle
        self._lock = asyncio.Lock()

        # Broadcast queues for all connected /ws/voice clients
        self._broadcast_queues: list[asyncio.Queue] = []

        # Audio pipeline objects (created lazily)
        self._audio_in = None
        self._vad = None
        self._wake = None

        # Asyncio queue bridging sync mic callback → async VAD loop
        self._mic_queue: asyncio.Queue | None = None
        self._vad_task: asyncio.Task | None = None

        # Config snapshot (updated by configure())
        self._cfg: dict[str, Any] = {}

        # PTT: mic open but VAD not used to end utterance
        self._ptt_active = False
        # Buffer that accumulates audio only while PTT is held down.
        # The VAD loop populates it; ptt_stop() drains it for transcription.
        self._ptt_buffer: list[bytes] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, cfg: dict[str, Any]) -> None:
        """Update voice config.  Applies immediately if already running."""
        self._cfg = cfg
        from backend.voice import stt as _stt
        _stt.configure(cfg.get("stt_model", "mlx-community/whisper-large-v3-turbo"))

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    def add_client(self, q: asyncio.Queue) -> None:
        self._broadcast_queues.append(q)

    def remove_client(self, q: asyncio.Queue) -> None:
        try:
            self._broadcast_queues.remove(q)
        except ValueError:
            pass

    def _emit(self, event: dict[str, Any]) -> None:
        for q in list(self._broadcast_queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def _set_state(self, state: VoiceState) -> None:
        if self._state != state:
            self._state = state
            self._emit({"type": "state", "state": state.value})

    @property
    def state(self) -> VoiceState:
        return self._state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start mic capture.  Mode (ptt/wakeword) from current config."""
        async with self._lock:
            if self._state != VoiceState.idle:
                logger.debug("start() called but state is %s — ignoring", self._state)
                return
            await self._open_mic()
            mode = self._cfg.get("activation_mode", "ptt")
            if mode == "wakeword":
                self._setup_wake()
                logger.info("Voice started in wake-word mode (model=%s)", self._cfg.get("wake_model"))
                self._set_state(VoiceState.listening)
            else:
                logger.info("Voice started in PTT mode")
                self._set_state(VoiceState.listening)  # waiting for PTT

    async def stop(self) -> None:
        """Tear down audio pipeline; return to idle."""
        async with self._lock:
            self._close_mic()
            self._set_state(VoiceState.idle)

    # ------------------------------------------------------------------
    # PTT controls
    # ------------------------------------------------------------------

    async def ptt_start(self) -> None:
        """Push-to-talk pressed — start capturing."""
        if self._state == VoiceState.idle:
            await self.start()
        self._ptt_buffer = []  # clear any leftover audio from a previous press
        self._ptt_active = True
        self._set_state(VoiceState.capturing)

    async def ptt_stop(self) -> None:
        """Push-to-talk released — end utterance, trigger transcription."""
        if self._state != VoiceState.capturing:
            return
        self._ptt_active = False
        utterance = b"".join(self._ptt_buffer)
        self._ptt_buffer = []
        if utterance:
            await self._transcribe(utterance)

    # ------------------------------------------------------------------
    # Internal: mic open/close
    # ------------------------------------------------------------------

    async def _open_mic(self) -> None:
        from backend.voice.audio_io import AudioInput

        loop = asyncio.get_event_loop()
        self._mic_queue = asyncio.Queue(maxsize=512)

        def _mic_callback(chunk: bytes) -> None:
            try:
                loop.call_soon_threadsafe(self._mic_queue.put_nowait, chunk)
            except Exception:  # noqa: BLE001
                pass

        device = self._cfg.get("mic_device") or None
        self._audio_in = AudioInput(device=device, callback=_mic_callback)
        try:
            self._audio_in.start()
        except RuntimeError as exc:
            self._emit({"type": "error", "message": str(exc)})
            raise

        self._vad_task = asyncio.create_task(self._vad_loop(), name="voice_vad_loop")

    def _close_mic(self) -> None:
        # 1. Signal the VAD loop to exit (sends sentinel before cancelling so
        #    the coroutine sees it and returns cleanly rather than via CancelledError).
        if self._mic_queue:
            try:
                self._mic_queue.put_nowait(None)
            except Exception:  # noqa: BLE001
                pass
        if self._vad_task and not self._vad_task.done():
            self._vad_task.cancel()
            self._vad_task = None
        # 2. Stop the PortAudio stream — no more callbacks will fire after this.
        if self._audio_in:
            self._audio_in.stop()
            self._audio_in = None
        self._mic_queue = None

    def _setup_wake(self) -> None:
        from backend.voice.wake import WakeWordDetector

        loop = asyncio.get_event_loop()

        def _on_wake() -> None:
            loop.call_soon_threadsafe(self._on_wake_detected)

        model_name = self._cfg.get("wake_model", "hey_otto")  # bundled at backend/voice/models/hey_otto.onnx
        self._wake = WakeWordDetector(model_name=model_name, threshold=0.5, on_wake=_on_wake)

    def _on_wake_detected(self) -> None:
        self._emit({"type": "wake"})
        if self._state == VoiceState.listening:
            self._set_state(VoiceState.capturing)

    # ------------------------------------------------------------------
    # VAD loop
    # ------------------------------------------------------------------

    async def _vad_loop(self) -> None:
        from backend.voice.vad import SileroVAD

        silence_secs = float(self._cfg.get("vad_silence_secs", 1.0))
        vad = SileroVAD(silence_secs=silence_secs)
        mode = self._cfg.get("activation_mode", "ptt")

        assert self._mic_queue is not None
        while True:
            try:
                chunk = await self._mic_queue.get()
            except asyncio.CancelledError:
                return
            if chunk is None:
                return

            # Wake-word and VAD inference are CPU-bound ONNX calls that run on
            # every ~32 ms mic chunk.  Running them inline here would block the
            # asyncio event loop ~30×/second, starving every HTTP handler (the
            # frontend's session/status/messages polling stalls behind it).
            # Offload the inference to a worker thread so the loop stays free;
            # all state mutation still happens on the loop after the await.
            if mode == "wakeword" and self._wake is not None:
                await asyncio.to_thread(self._wake.process_chunk, chunk)

            if self._state == VoiceState.capturing:
                if mode == "ptt":
                    # Only accumulate audio while the PTT button is actively held.
                    # ptt_stop() reads self._ptt_buffer for transcription.
                    if self._ptt_active:
                        self._ptt_buffer.append(chunk)
                else:
                    result = await asyncio.to_thread(vad.process_chunk, chunk)
                    if result is not None:
                        asyncio.create_task(self._transcribe(result))
                        self._set_state(VoiceState.listening)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def _transcribe(self, pcm_bytes: bytes) -> None:
        self._set_state(VoiceState.transcribing)
        try:
            from backend.voice import stt as _stt
            language = self._cfg.get("stt_language") or None
            text = await _stt.transcribe(pcm_bytes, language=language)
            if text.strip():
                self._emit({"type": "transcript", "text": text})
        except Exception as exc:  # noqa: BLE001
            logger.warning("STT error: %s", exc)
            self._emit({"type": "error", "message": str(exc)})
        finally:
            mode = self._cfg.get("activation_mode", "ptt")
            if self._state == VoiceState.transcribing:
                if mode == "wakeword":
                    self._set_state(VoiceState.listening)
                else:
                    # PTT: release the mic so the OS mic indicator turns off
                    self._close_mic()
                    self._set_state(VoiceState.idle)


# Module-level singleton — created once on first import
_manager = VoiceManager()


def get_manager() -> VoiceManager:
    return _manager

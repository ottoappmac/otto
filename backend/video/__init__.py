"""Video understanding subsystem.

Lets Otto watch videos on the user's behalf — uploaded files, screen
recordings, and online (YouTube) videos.  Two provider paths:

- **Gemini (native)** — the clip or YouTube URL is sent straight to
  Gemini, which samples frames + audio itself (``backend.video.gemini``).
- **Frame-based** — Otto samples frames with ffmpeg and transcribes the
  audio with the on-device Whisper stack, then hands a normal multimodal
  message to any vision-capable model (``backend.video.ingest``).
"""

"""Transcript parsers for various agent platforms."""

from tools.transcripts.parsers.base import (
    ModelCall,
    ProjectInfo,
    SessionInfo,
    SessionSummary,
    ToolCallRecord,
    TranscriptParser,
    Turn,
)

__all__ = [
    "ModelCall",
    "ProjectInfo",
    "SessionInfo",
    "SessionSummary",
    "ToolCallRecord",
    "TranscriptParser",
    "Turn",
]


def get_parser(platform: str, **kwargs: object) -> TranscriptParser:
    """Convenience factory — returns the parser for *platform*.

    Lazy-imports platform modules to keep startup fast.
    Extra ``kwargs`` are forwarded to the parser constructor (used by
    OpenClaw to pass SSH configuration).
    """
    if platform == "claude_code":
        from tools.transcripts.parsers.claude_code import ClaudeCodeParser
        return ClaudeCodeParser()
    if platform == "cowork":
        from tools.transcripts.parsers.cowork import CoworkParser
        return CoworkParser()
    if platform == "openclaw":
        from tools.transcripts.parsers.openclaw import OpenClawParser
        return OpenClawParser(**kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"Unknown platform: {platform!r}. "
        f"Available: claude_code, cowork, openclaw"
    )

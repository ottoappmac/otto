"""Schemas"""
from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass
from PIL import Image
import json


class BBox(TypedDict):
    """BBox"""
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str
    src: Optional[str]
    numerical_label: Optional[str]


class ScreenshotImage(BaseModel):
    """ScreenshotImage"""
    image: Image.Image
    is_new_window: bool = False
    window_selector: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AnnotatedImage(BaseModel):
    """AnnotatedImage"""
    raw_img: str
    img: str
    bboxes: Optional[List[BBox]] = Field(default=[])
    img_width: int
    img_height: int
    model_config = ConfigDict(arbitrary_types_allowed=True)


class VisualAnnotatedImage(AnnotatedImage):
    """Annotated image with window context for visual/screenshot-based interaction."""
    window_selector: Optional[str] = None
    is_new_window: Optional[bool] = None


class Action(TypedDict):
    """Action"""
    action: str
    args: Optional[List[str]]
    log: Optional[str]
    actioned: Optional[bool]


@dataclass(kw_only=True, frozen=True)
class ToolResult:
    output: str | None = None
    error: str | None = None
    system: str | None = None
    successful: bool = True
    name: str | None = None
    transaction_message: str | None = None


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""
    def __init__(self, *, output=None, error=None, system=None, name=None):
        super().__init__(output=output, error=error, system=system, successful=False, name=None)


class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


@dataclass(kw_only=True, frozen=True)
class ImageToolResult(ToolResult):
    base64_image: str
    is_new_window: bool = False
    window_selector: str | None = None


@dataclass(kw_only=True)
class VisualResult:
    execution_result: str
    successful: bool
    script: Optional[str] = None
    visionscript: Optional[str] = None  # deprecated alias for script
    description: Optional[str] = None
    action_mode: Optional[Literal['input', 'verify', 'buffer']] = None
    buffers: Optional[dict[str, str]] = None

    def json_str(self):
        return json.dumps({k: v for k, v in self.__dict__.items() if v is not None})


@dataclass(kw_only=True, frozen=True)
class ScriptToolResult(ToolResult):
    action_mode: str
    description: str
    script: str
    successful: bool = True
    buffers: dict[str, str] | None = None


def build_visual_tool_result(visual_result: VisualResult) -> ToolResult:
    if visual_result.successful:
        script_content = visual_result.script or visual_result.visionscript
        if script_content:
            return ScriptToolResult(
                output=visual_result.execution_result,
                action_mode=visual_result.action_mode or "none",
                description=visual_result.description or visual_result.execution_result,
                script=script_content,
                successful=visual_result.successful,
                buffers=visual_result.buffers,
            )
        return ToolResult(output=visual_result.execution_result)
    return ToolFailure(error=visual_result.execution_result)


# Backwards-compatible aliases
VisionAIAnnotatedImage = VisualAnnotatedImage
VisionAiResult = VisualResult
VisionscriptToolResult = ScriptToolResult
build_vai_tool_result = build_visual_tool_result

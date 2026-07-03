from typing import Literal, TypedDict, Mapping, Union, Optional, get_args, cast
from strenum import StrEnum
from dataclasses import dataclass, field

import os
import time

from PIL import Image

from anthropic.types.beta import BetaToolUnionParam

from tools.base import Navigator
from tools.anthropic.base import BaseAnthropicTool
from tools.schemas import ToolError, ToolResult, ImageToolResult, VisualResult
from tools.image_utils import resize_image_async, image_to_base64_async


@dataclass(frozen=True)
class Dimension:
    x: int
    y: int


class ClaudeDimensions:
    # sizes above XGA/WXGA are not recommended for Claude
    # scale down and/or pad to one of these targets
    target_resolutions = [Dimension(1024, 768), Dimension(1280, 800), Dimension(1366, 768)]

    def __init__(self, source_width: int, source_height: int):
        source_width = max(1, source_width)
        source_height = max(1, source_height)
        self.source = Dimension(source_width, source_height)
        self.claude = min(self.target_resolutions, key=lambda res: abs((res.x / res.y) - (source_width / source_height)))
        self.target = self.source
        if self.target.x > self.claude.x:
            self.target = Dimension(self.claude.x, int(source_height * self.claude.x / source_width))
        if self.target.y > self.claude.y:
            self.target = Dimension(int(source_width * self.claude.y / source_height), self.claude.y)
        self.offset = Dimension((self.claude.x - self.target.x) // 2, (self.claude.y - self.target.y) // 2)

    @property
    def resize_required(self) -> bool:
        return self.source.x != self.claude.x or self.source.y != self.claude.y

    def unscale_coords(self, x: int, y: int) -> tuple[int, int]:
        scalex = self.source.x / self.target.x
        scaley = self.source.y / self.target.y
        return round((x - self.offset.x) * scalex), round((y - self.offset.y) * scaley)


class Resolution(TypedDict):
    width: int
    height: int


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "scroll"
]


@dataclass(frozen=True)
class ActionConstants():
    NON_KEY_ACTIONS: tuple[str, ...] = (
        "left_click", "right_click", "double_click", "middle_click", "mouse_move", "left_click_drag")
    KEY_ACTIONS: tuple[str, ...] = ("key", "type")
    MAX_SCALING_TARGETS: Mapping[str, Resolution] = field(
        default_factory=lambda: {
            "XGA": Resolution(width=1024, height=768),
            "WXGA": Resolution(width=1280, height=800),
            "FWXGA": Resolution(width=1366, height=768),
        }
    )
    TYPING_DELAY_MS: int = 12
    TYPING_GROUP_SIZE: int = 50


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


ScrollDirection = Literal["up", "down", "left", "right"]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20250124"] = "computer_20250124"
    width: int
    height: int
    display_num: int | None
    display_prefix: str
    target_dimension: Resolution

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        return {
            "display_width_px": self.claude_dimensions.claude.x,
            "display_height_px": self.claude_dimensions.claude.y,
            "display_number": self.display_num,
        }

    def to_params(self):
        return cast(BetaToolUnionParam, {"name": self.name, "type": self.api_type, **self.options})

    def __init__(self, navigator: Navigator):
        super().__init__()
        self.claude_dimensions: ClaudeDimensions = ClaudeDimensions(1024, 768)
        self.navigator = navigator

        if (display_num := os.getenv("DISPLAY_NUM")) is not None:
            self.display_num = int(display_num)
            self._display_prefix = f"DISPLAY=:{self.display_num} "
        else:
            self.display_num = None
            self._display_prefix = ""

    def _unscale_coordinates(self, x: int, y: int):
        """Unscale coordinates from the scaled-down resolution back to the original resolution."""
        if not self._scaling_enabled:
            return x, y

        if self.target_dimension is None:
            return x, y

        x_unscaling_factor = self.width / self.target_dimension["width"]
        y_unscaling_factor = self.height / self.target_dimension["height"]
        return round(x * x_unscaling_factor), round(y * y_unscaling_factor)

    def _get_coordinates(self, param: tuple[int, int] | None) -> tuple[int, int]:
        if param is None:
            raise ToolError("`coordinate` is required for this action.")
        if len(param) != 2 or not all(isinstance(coord, int) for coord in param):
            raise ToolError(f"`coordinate` must be a tuple of two integers, got: {param}")
        return self.claude_dimensions.unscale_coords(param[0], param[1])

    async def screenshot(self) -> ImageToolResult:
        """Take a screenshot of the current screen"""
        if screenshot := await self.navigator.screenshot():
            if not screenshot.image:
                raise ToolError("Failed to take screenshot")
            dim = ClaudeDimensions(screenshot.image.width, screenshot.image.height)
            if dim.resize_required:
                resized = await resize_image_async(screenshot.image, dim.target.x, dim.target.y)
                img = Image.new("RGB", (dim.claude.x, dim.claude.y), (0, 0, 0))
                img.paste(resized, (dim.offset.x, dim.offset.y))
            else:
                img = screenshot.image
            self.claude_dimensions = dim
            result = ImageToolResult(
                output="Screenshot taken",
                base64_image=await image_to_base64_async(img),
                is_new_window=screenshot.is_new_window,
                window_selector=screenshot.window_selector)
            return result
        raise ToolError("Failed to take screenshot")

    def _validate_non_key_action(self, action: Action, text: str | None, coordinate: tuple[int, int] | None):
        if text is not None:
            raise ToolError(f"Text is not accepted for '{action}' action.")
        if coordinate is None:
            raise ToolError(f"Coordinate is required for '{action}' action.")

    async def _execute_non_key_action(
            self, action: Action, coordinate: Optional[tuple[int, int]]) -> Union[
                VisualResult, ToolResult, ImageToolResult, ToolError]:
        tool_result = ToolResult()
        action_map = {
            "left_click": self.navigator.click,
            "right_click": self.navigator.right_click,
            "middle_click": self.navigator.middle_click,
            "double_click": self.navigator.double_click,
            "mouse_move": self.navigator.move
        }

        if action == "screenshot":
            tool_result = await self.screenshot()
        else:
            if action in action_map and coordinate:
                x, y = self._get_coordinates(coordinate)
                tool_result = await action_map[action](x, y)
        return tool_result

    def _validate_key_action(self, action: Action, text: str | None, coordinate: tuple[int, int] | None):
        if text is None:
            raise ToolError(f"Text is required for '{action}' action.")
        if not isinstance(text, str):
            raise ToolError(f"Text must be a string, got: {text}.")

    async def _execute_key_action(
            self,
            action: Action,
            coordinate: Optional[tuple[int, int]], text: str) -> Union[ToolResult, ImageToolResult, ToolError]:
        tool_result = ToolResult()
        action_map = {
            "key": self.navigator.key_press,
            "type": self.navigator.type_text
        }

        if action == "screenshot":
            tool_result = await self.screenshot()
        else:
            if action in action_map:
                x = None
                y = None
                if coordinate:
                    x, y = self._get_coordinates(coordinate)
                tool_result = await action_map[action](x, y, text)
        return tool_result

    async def scroll(self, coordinate: tuple[int, int], scroll_direction: ScrollDirection, scroll_amount: int):
        tool_result = ToolResult()
        if scroll_direction is None or scroll_direction not in get_args(ScrollDirection):
            raise ToolError(
                f"{scroll_direction=} must be 'up', 'down', 'left', or 'right'"
            )
        if not isinstance(scroll_amount, int) or scroll_amount < 0:
            raise ToolError(f"{scroll_amount=} must be a non-negative int")

        x, y = self._get_coordinates(coordinate)
        tool_result = await self.navigator.scroll(x, y, scroll_direction, scroll_amount)
        return tool_result

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        scroll_direction: ScrollDirection | None = None,
        scroll_amount: int | None = None,
        **kwargs,
    ):
        tool_result = None

        if action == "screenshot":
            tool_result = await self._execute_non_key_action(action, coordinate)
        elif action == "scroll":
            self._validate_non_key_action(action, text, coordinate)
            tool_result = await self.scroll(coordinate, scroll_direction, scroll_amount)
        elif action in ActionConstants.NON_KEY_ACTIONS:
            self._validate_non_key_action(action, text, coordinate)
            tool_result = await self._execute_non_key_action(action, coordinate)
        elif action in ActionConstants.KEY_ACTIONS:
            self._validate_key_action(action, text, coordinate)
            if text:
                tool_result = await self._execute_key_action(action, coordinate, text)
        else:
            raise ToolError(f"Invalid action: {action}")

        return tool_result

    def track_action_time(self):
        self.last_action_time = time.time()

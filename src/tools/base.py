from abc import ABC, abstractmethod
from typing import Literal, Union
from pydantic import BaseModel, ConfigDict
from tools.schemas import ScreenshotImage, AnnotatedImage, VisualAnnotatedImage, ToolResult, VisualResult

ToolResponse = Union[ToolResult, VisualResult]


class Navigator(ABC, BaseModel):
    """Navigator"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def connect(self) -> object:
        ...

    @abstractmethod
    async def get_current_tab(self) -> object:
        ...

    @abstractmethod
    async def go_to(self, url: str) -> ToolResponse:
        ...

    @abstractmethod
    async def go_back(self) -> ToolResponse:
        ...

    @abstractmethod
    async def screenshot(self) -> ScreenshotImage:
        ...

    @abstractmethod
    async def annotate(self) -> AnnotatedImage | VisualAnnotatedImage:
        ...

    @abstractmethod
    async def move(self, x: int, y: int) -> ToolResponse:
        ...

    @abstractmethod
    async def click(self, x: int, y: int) -> ToolResponse:
        ...

    @abstractmethod
    async def right_click(self, x: int, y: int) -> ToolResponse:
        ...

    @abstractmethod
    async def middle_click(self, x: int, y: int) -> ToolResponse:
        ...

    @abstractmethod
    async def double_click(self, x: int, y: int) -> ToolResponse:
        ...

    @abstractmethod
    async def key_press(self, x: int, y: int, key: str) -> ToolResponse:
        ...

    @abstractmethod
    async def type_text(self, x: int, y: int, content: str) -> ToolResponse:
        ...

    @abstractmethod
    async def delete_text(self, x: int, y: int) -> ToolResponse:
        ...

    @abstractmethod
    async def type_text_to_table(self, x: int, y: int, table_row_id: str,
                                 col_header_name: str, content: str) -> ToolResponse:
        ...

    @abstractmethod
    async def scroll(
            self, x: int, y: int, direction: Literal['up', 'down', 'left', 'right'], amount: int) -> ToolResponse:
        ...

    @abstractmethod
    async def search(self, content: str) -> ToolResponse:
        ...

    @abstractmethod
    async def select_dropdown(self, x: int, y: int, content: str) -> ToolResponse:
        ...

    @abstractmethod
    async def check_checkbox(self, x: int, y: int) -> ToolResponse:
        ...

    @abstractmethod
    async def uncheck_checkbox(self, x: int, y: int) -> ToolResponse:
        ...

    @abstractmethod
    async def click_table_radiobutton(self, x: int, y: int, table_row_id: str) -> ToolResponse:
        ...

    @abstractmethod
    async def check_table_checkbox(self, x: int, y: int, table_row_id: str) -> ToolResponse:
        ...

    @abstractmethod
    async def uncheck_table_checkbox(self, x: int, y: int, table_row_id: str) -> ToolResponse:
        ...

    @abstractmethod
    async def click_table_cell(self, x: int, y: int, table_row_id: str, col_header_name: str) -> ToolResponse:
        ...

    @abstractmethod
    async def ask_knowledge_base(self, query: str) -> ToolResponse:
        ...

    @abstractmethod
    def get_navigator_instructions(self) -> str:
        ...

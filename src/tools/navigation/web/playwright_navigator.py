"""Claude Navigators"""
from typing import Literal, Optional
from pydantic import Field
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from langchain_classic.chains.base import Chain

from tools.navigation.web.schemas import ScreenshotImage, AnnotatedImage, ToolResult
from tools.navigation.base import Navigator
from tools.navigation.web.playwright_tools import (
    screenshot as _screenshot,
    annotate_page,
    get_current_browser_tab,
    go_to_web_page,
    move_by_coordinates,
    click_by_coordinates,
    web_search,
    right_click_by_coordinates,
    middle_click_by_coordinates,
    double_click_by_coordinates,
    keypress_by_coordinates,
    type_by_coordinates,
    scroll_by_coordinates
)


class PlaywrightComputerUseNavigator(Navigator):
    """PlaywrightComputerUseNavigator"""
    width: Optional[int]
    height: Optional[int]
    headless_flag: bool
    knowledge_base_rag: Optional[Chain] = Field(default=None)

    browser: Optional[Browser] = Field(default=None)
    browser_context: Optional[BrowserContext] = Field(default=None)
    page: Optional[Page] = Field(default=None)

    async def connect(self) -> Page:
        args = []
        args.append("--start-maximized")
        if self.width and self.height:
            args.append(f"--window-size={self.width},{self.height}")

        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=self.headless_flag, args=args
        )
        self.browser_context = await self.browser.new_context(no_viewport=True if self.headless_flag else False)
        self.page = await self.browser_context.new_page()
        return self.page

    async def get_current_tab(self) -> Page:
        if not self.page:
            await self.connect()
        self.page = await get_current_browser_tab.ainvoke({"browser_context": self.browser_context})
        if self.page:
            return self.page
        raise ValueError("page is not set")

    async def go_to(self, url: str) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await go_to_web_page.ainvoke({"url": url, "page": self.page})
        return res

    async def screenshot(self) -> ScreenshotImage:
        self.page = await self.get_current_tab()
        res = await _screenshot.ainvoke({"page": self.page})
        return res

    async def annotate(self) -> AnnotatedImage:
        self.page = await self.get_current_tab()
        res = await annotate_page.ainvoke({"page": self.page})
        return res

    async def move(self, x: int, y: int) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await move_by_coordinates.ainvoke({"x": x, "y": y, "page": self.page})
        return res

    async def click(self, x: int, y: int) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await click_by_coordinates.ainvoke({"x": x, "y": y, "page": self.page})
        return res

    async def right_click(self, x: int, y: int) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await right_click_by_coordinates.ainvoke({"x": x, "y": y, "page": self.page})
        return res

    async def middle_click(self, x: int, y: int) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await middle_click_by_coordinates.ainvoke({"x": x, "y": y, "page": self.page})
        return res

    async def double_click(self, x: int, y: int) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await double_click_by_coordinates.ainvoke({"x": x, "y": y, "page": self.page})
        return res

    async def key_press(self, x: int, y: int, key: str) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await keypress_by_coordinates.ainvoke({"x": x, "y": y, "key": key, "page": self.page})
        return res

    async def type_text(self, x: int, y: int, content: str) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await type_by_coordinates.ainvoke({"x": x, "y": y, "content": content, "page": self.page})
        return res

    async def delete_text(self, x: int, y: int) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await type_by_coordinates.ainvoke({"x": x, "y": y, "content": "", "page": self.page})
        return res

    async def scroll(self, x: int, y: int, direction: Literal['up', 'down', 'left', 'right'], amount: int) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await scroll_by_coordinates.ainvoke(
            {"x": x, "y": y, "direction": direction, "amount": amount if amount >= 150 else 150, "page": self.page})
        return res

    async def search(self, content: str) -> ToolResult:
        self.page = await self.get_current_tab()
        res = await web_search.ainvoke(
            {"content": content, "page": self.page})
        return res

    async def ask_knowledge_base(self, query: str) -> ToolResult:
        answer = ""
        system = f"Asking Knowledge Base query: {query}"
        if self.knowledge_base_rag:
            input_key = self.knowledge_base_rag.input_keys[0]
            answer = await self.knowledge_base_rag.ainvoke(input={input_key: query})
            return ToolResult(output=str(answer), system=system)
        return ToolResult(error=f"Cannot asking Knowledge Base query: {query}.", system=system)

    async def go_back(self) -> ToolResult:
        ...

    async def type_text_to_table(self, x: int, y: int,  table_row_id: str,
                                 col_header_name: str, content: str) -> ToolResult:
        ...

    async def select_dropdown(self, x: int, y: int, content: str) -> ToolResult:
        ...

    async def check_checkbox(self, x: int, y: int) -> ToolResult:
        ...

    async def uncheck_checkbox(self, x: int, y: int) -> ToolResult:
        ...

    async def click_table_radiobutton(self, x: int, y: int, table_row_id: str) -> ToolResult:
        ...

    async def check_table_checkbox(self, x: int, y: int, table_row_id: str) -> ToolResult:
        ...

    async def uncheck_table_checkbox(self, x: int, y: int, table_row_id: str) -> ToolResult:
        ...

    async def click_table_cell(self, x: int, y: int, table_row_id: str, col_header_name: str) -> ToolResult:
        ...

    def get_navigator_instructions(self) -> str:
        return (
            "Key Press Guidelines:\n"
            "   - Character Keys: 'a', 'b', 'c', ..., 'z', 'A', 'B', ..., 'Z', '0', '1', ..., '9'\n"
            "   - Modifier Keys: 'Shift', 'Control', 'Alt', 'Meta' (Windows Key / Command Key)\n"
            "   - Arrow Keys: 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'\n"
            "   - Function Keys: 'F1', 'F2', ..., 'F12'\n"
            "   - Navigation Keys: 'Enter', 'Escape', 'Backspace', 'Tab', 'Delete', 'Insert',"
            " 'Home', 'End', 'PageUp', 'PageDown'\n"
            "   - Numpad Keys: 'Numpad0', 'Numpad1', ..., 'Numpad9', 'NumpadAdd', 'NumpadSubtract',"
            " 'NumpadMultiply', 'NumpadDivide', 'NumpadDecimal', 'NumpadEnter'\n"
            "   - Symbol Keys: 'Space', 'Minus', 'Equal', 'BracketLeft', 'BracketRight', 'Backslash',"
            " 'Semicolon', 'Quote', 'Comma', 'Period', 'Slash', 'Backquote'\n"
            "   - Media Keys (if supported): 'AudioVolumeMute', 'AudioVolumeDown', 'AudioVolumeUp',"
            " 'MediaTrackNext', 'MediaTrackPrevious', 'MediaStop', 'MediaPlayPause'\n\n"

            "   - Examples:\n"
            "       - Pressing a Single Key: To press the Return or Enter key, return 'Enter'"
            "       - Pressing a Key Combination: e.g. To simulate Control S key presses, return 'Ctrl+S'\n"
            "       - Simulating Numpad Keys: e.g To key press the Numpad 5 key, return 'Numpad5'\n"
        )

    @classmethod
    async def from_params(
            cls,
            knowledge_base_rag: Optional[Chain] = None,
            width: Optional[int] = 0,
            height: Optional[int] = 0,
            headless_flag: bool = False):
        # Initialize Playwright asynchronously
        return cls(
            knowledge_base_rag=knowledge_base_rag,
            width=width,
            height=height,
            headless_flag=headless_flag)

""" Playwright Toolkit """
from typing import Dict, Any, Annotated, Literal, Optional, cast
from io import BytesIO
from langchain_core.tools import InjectedToolArg, tool

import os
import platform
import base64
import asyncio
from urllib.parse import quote_plus
from PIL import Image
from playwright.async_api import Page
from playwright.async_api._generated import BrowserContext
from langchain_core.runnables import chain as chain_decorator
from tools.navigation.web.schemas import ToolResult, ScreenshotImage, AnnotatedImage


@chain_decorator
async def mark_page(page: Page) -> Dict[str, Any]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    js_file_path = os.path.join(current_dir, "mark_page.js")

    with open(js_file_path, encoding="utf-8") as f:
        mark_page_script = f.read()

    raw_screenshot = await page.screenshot()

    await page.evaluate(mark_page_script)

    bboxes = None  # Initialize bboxes to avoid unbound error
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            # May be loading...
            await asyncio.sleep(3)

    if bboxes is None:
        raise RuntimeError("Failed to get bounding boxes after multiple attempts")

    annotated_screenshot = await page.screenshot()

    # Ensure the bboxes don't follow us around
    await page.evaluate("unmarkPage()")

    viewport_size = page.viewport_size

    return {
        "raw_img": base64.b64encode(raw_screenshot).decode(),
        "img": base64.b64encode(annotated_screenshot).decode(),
        "bboxes": bboxes,
        "img_width": viewport_size["width"] if viewport_size else 0,
        "img_height": viewport_size["height"] if viewport_size else 0
    }


@tool("get-current-browser-tab")
async def get_current_browser_tab(browser_context: Annotated[BrowserContext, InjectedToolArg]) -> Page:
    """Get the current browser tab."""
    pages = browser_context.pages

    if not pages:
        raise RuntimeError("No pages found in the browser context")

    if len(pages) > 1:
        page = pages[-1]
        await page.bring_to_front()
    else:
        page = pages[0]

    return page


@tool("go-to-web-page")
async def go_to_web_page(url: Annotated[str, "Web page URL"], page: Annotated[Page, InjectedToolArg]) -> ToolResult:
    """A tool that navigates to a Web Page url"""
    output: str = ""
    error: str = ""

    try:
        await page.goto(url)
        output = f"Navigated to page {url}."
    except Exception as e:
        error = f"Could not navigate to {url} \n{e}\n"

    return ToolResult(output=output, error=error, system=f"Navigate to {url}")


@tool("screenshot")
async def screenshot(page: Annotated[Page, InjectedToolArg]) -> ScreenshotImage:
    """A tool that annotates the detected elements on a Web Page"""
    screenshot_image = None
    try:
        screenshot_bytes = await page.screenshot()
        screenshot_image = Image.open(BytesIO(screenshot_bytes))
    except Exception:
        raise ValueError("Unable to take screenshot")
    return ScreenshotImage(image=screenshot_image)


@tool("annotate-page")
async def annotate_page(page: Annotated[Page, InjectedToolArg]) -> AnnotatedImage:
    """A tool that annotates the detected elements on a Web Page"""
    marked_page_dict = await mark_page.with_retry().ainvoke(page)
    if not marked_page_dict.get("raw_img"):
        marked_page_dict["img"] = ""
    if not marked_page_dict.get("img"):
        marked_page_dict["img"] = ""
    if not marked_page_dict.get("bboxes"):
        marked_page_dict["bboxes"] = []

    return AnnotatedImage(**marked_page_dict)


@tool("move-by-coordinates")
async def move_by_coordinates(
    x: Annotated[int, "x-coordinate"],
    y: Annotated[int, "y-coordinate"],
    page: Annotated[
        Page, InjectedToolArg]) -> ToolResult:
    """A tool that mouse moves on a Web Page to x, y coordinates"""
    output: str = ""
    error: str = ""
    try:
        await page.mouse.move(x, y)
        output = f"Mouse moved to x: {x}, y: {y} successfully."
    except Exception as e:
        error = f"Error: Could not mouse move to x: {x}, y: {y}: {e}"

    return ToolResult(output=output, error=error, system=f"Mouse Move to x: {x}, y: {y}")


@tool("click-by-coordinates")
async def click_by_coordinates(
    x: Annotated[int, "x-coordinate"],
    y: Annotated[int, "y-coordinate"],
    page: Annotated[
        Page, InjectedToolArg]) -> ToolResult:
    """A tool that clicks on a Web Page based on x, y coordinates"""
    output: str = ""
    error: str = ""
    try:
        await page.mouse.click(x, y, button="left")
        output = f"Clicked Element at x: {x}, y: {y} successfully."
    except Exception as e:
        error = f"Error: Could not click element at x: {x}, y: {y}: {e}"

    return ToolResult(output=output, error=error, system=f"Click on x: {x}, y: {y}")


@tool("right-click-by-coordinates")
async def right_click_by_coordinates(
    x: Annotated[int, "x-coordinate"],
    y: Annotated[int, "y-coordinate"],
    page: Annotated[
        Page, InjectedToolArg]) -> ToolResult:
    """A tool that mouse right clicks on a Web Page based on x, y coordinates"""
    output: str = ""
    error: str = ""
    try:
        await page.mouse.click(x, y, button="right")
        output = f"Right clicked Element at x: {x}, y: {y} successfully."
    except Exception as e:
        error = f"Error: Could not right click element at x: {x}, y: {y}: {e}"

    return ToolResult(output=output, error=error, system=f"Right Click on x: {x}, y: {y}")


@tool("middle-click-by-coordinates")
async def middle_click_by_coordinates(
    x: Annotated[int, "x-coordinate"],
    y: Annotated[int, "y-coordinate"],
    page: Annotated[
        Page, InjectedToolArg]) -> ToolResult:
    """A tool that mouse middle button clicks on a Web Page based on x, y coordinates"""
    output: str = ""
    error: str = ""
    try:
        await page.mouse.click(x, y, button="middle")
        output = f"Middle clicked Element at x: {x}, y: {y} successfully."
    except Exception as e:
        error = f"Error: Could not middle click element at x: {x}, y: {y}: {e}"

    return ToolResult(output=output, error=error, system=f"Middle Click on x: {x}, y: {y}")


@tool("double-click-by-coordinates")
async def double_click_by_coordinates(
    x: Annotated[int, "x-coordinate"],
    y: Annotated[int, "y-coordinate"],
    page: Annotated[
        Page, InjectedToolArg]) -> ToolResult:
    """A tool that double clicks on a Web Page based on x, y coordinates"""
    output: str = ""
    error: str = ""
    try:
        await page.mouse.dblclick(x, y)
        output = f"Double clicked Element at x: {x}, y: {y} successfully."
    except Exception as e:
        error = f"Error: Could not double click element at x: {x}, y: {y}: {e}"

    return ToolResult(output=output, error=error, system=f"Double Click on x: {x}, y: {y}")


@tool("type-by-coordinates")
async def type_by_coordinates(
    x: Optional[Annotated[int, "x-coordinate"]],
    y: Optional[Annotated[int, "y-coordinate"]],
    content: Annotated[str, "content"],
    page: Annotated[
        Page, InjectedToolArg]) -> ToolResult:
    """A tool that types content into a Web Page based on x, y coordinates"""
    output: str = ""
    error: str = ""
    try:
        if x and y:
            await page.mouse.click(x, y)
        select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
        await page.keyboard.press(select_all)
        await page.keyboard.press("Backspace")
        await page.keyboard.type(content)

        output = f"Typed {content} into x: {x if x else ''}, y: {y if y else ''} successfully."
    except Exception as e:
        error = f"Error: Could not type {content} into x: {x}, y: {y}: {e}"

    return ToolResult(output=output, error=error, system=f"Type {content} " + (f"into x: {x}, y: {y}" if x and y else ""))


# Map common key aliases to Playwright key names (https://playwright.dev/python/docs/api/class-keyboard)
_KEY_ALIASES = {"Return": "Enter", "Return/Enter": "Enter"}


@tool("keypress")
async def keypress_by_coordinates(
    x: Optional[Annotated[int, "x-coordinate"]],
    y: Optional[Annotated[int, "y-coordinate"]],
    key: Annotated[str, "Key to press."],
    page: Annotated[
        Page, InjectedToolArg]) -> ToolResult:
    """A tool that actions a keypress"""
    output: str = ""
    error: str = ""
    try:
        key = _KEY_ALIASES.get(key, key)
        if x and y:
            await page.mouse.click(x, y)
        await page.keyboard.press(key)

        output = f"Keypress {key} successful."
    except Exception as e:
        error = f"Error: Could not keypress {key}: {e}"

    return ToolResult(output=output, error=error, system=f"Keypress {key} " + (f"into x: {x}, y: {y}" if x and y else ""))


@tool("scroll")
async def scroll_by_coordinates(
    x: Annotated[int, "x-coordinate"],
    y: Annotated[int, "y-coordinate"],
    direction: Annotated[Literal["up", "down", "left", "right"], "Scroll direction [up, down, left, right]"],
    amount: Annotated[int, "Scroll amount"],
    page: Annotated[
        Page, InjectedToolArg]) -> ToolResult:
    """A tool that types content into a Web Page based on x, y coordinates"""
    output: str = ""
    error: str = ""
    signed_scroll_amount = 0
    delta_dict = {
        "delta_x": 0,
        "delta_y": 0,
    }
    try:
        if direction.lower() in ("up", "left"):
            signed_scroll_amount = -amount
        else:
            signed_scroll_amount = amount
        await page.mouse.move(x, y)
        if direction.lower() in ("up", "down"):
            delta_dict["delta_x"] = 0
            delta_dict["delta_y"] = signed_scroll_amount
        else:
            delta_dict["delta_x"] = signed_scroll_amount
            delta_dict["delta_y"] = 0

        await page.mouse.wheel(**delta_dict)

        output = f"Scrolled {direction} by {amount} at x: {x}, y: {y}."
    except Exception as e:
        error = f"Error: Could not scroll {direction} by {amount} at into x: {x}, y: {y}: {e}"

    return ToolResult(output=output, error=error, system=f"Scroll {direction} by {amount} at x: {x}, y: {y}.")


@tool("web-search")
async def web_search(
    content: Annotated[str, "Search term"],
    page: Annotated[
        Page, InjectedToolArg]) -> ToolResult:
    """A tool that types content into a Web Page based on x, y coordinates"""
    output: str = ""
    error: str = ""

    try:
        content = quote_plus(content)
        # Use DuckDuckGo to avoid reCAPTCHA; Google often blocks automated browsers
        res = await go_to_web_page.ainvoke({"url": f"https://duckduckgo.com/?q={content}", "page": page})
        res = cast(ToolResult, res)
        output = res.output if res.output else ""
    except Exception as e:
        error = f"Error: Could not search the web for {content}: {e}"

    return ToolResult(output=output, error=error, system=f"Search {content} on the web.")

"""Image Utilities"""
import base64
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix='screenshot')


def _img_to_BytesIO(img: Image.Image, format: str = "png") -> BytesIO:
    imgdata = BytesIO()
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(imgdata, format=format)
    return imgdata


def image_to_base64(img: Image.Image, format: str = "png") -> str:
    return base64.b64encode(_img_to_BytesIO(img, format).getvalue()).decode('utf-8')


async def image_to_base64_async(image: Image.Image, format: str = "png") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, image_to_base64, image, format)


async def resize_image_async(img: Image.Image, width: int, height: int) -> Image.Image:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, functools.partial(img.resize, (width, height)))

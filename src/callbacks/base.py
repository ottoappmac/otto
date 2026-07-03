from typing import Dict, Any, Awaitable, Callable, Literal, Optional, cast
from enum import Enum
import inspect
import logging

from langchain_core.callbacks import AsyncCallbackHandler

logger = logging.getLogger(__name__)


class MessageType(Enum):
    Info = "info"
    Warning = "warning"
    Error = "error"
    Code = "code"


class EncodedImage:
    def __init__(self, image_data: bytes, type: Literal['jpeg', 'png', 'webp']) -> None:
        self.image_data = image_data
        self.type = type


TestCaseStatus = Literal['start', 'stop', 'pause', 'resume']


class WebCallbackHandler(AsyncCallbackHandler):
    """Asynchronous callback for manually validating values."""
    def __init__(
        self,
        image_sender: Optional[Callable[[EncodedImage], Awaitable[None]]] = None,
        messenger: Optional[Callable[..., Awaitable[None]]] = None,
        user_input: Optional[Callable[[str], Awaitable[str]] | Callable[[str], str]] = None,
        agent_logger: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        status_logger: Optional[Callable[[str], Awaitable[None]]] = None,
        anonymization_base_url: Optional[str] = None,
    ):
        self.image_sender = image_sender
        self.messenger = messenger
        self.user_input = user_input
        self.agent_logger = agent_logger
        self.status_logger = status_logger

    async def on_message(
        self,
        input_str: str,
        message_type: MessageType = MessageType.Info,
        use_anonymizer_service: bool = False
    ) -> None:
        if self.messenger:
            try:
                await self.messenger(input_str, message_type, use_anonymizer_service)
            except TypeError:
                await self.messenger(input_str, message_type)

    async def on_image(self, image: EncodedImage) -> None:
        if self.image_sender:
            await self.image_sender(image)

    async def on_user_input(self, question: str) -> str | None:
        if self.user_input:
            if inspect.iscoroutinefunction(self.user_input):
                return await self.user_input(question)
            else:
                cast(Callable[[str], str], self.user_input)(question)
        return None

    async def on_agent_logs(self, agent_logs: Dict[str, Any]) -> None:
        if self.agent_logger:
            if inspect.iscoroutinefunction(self.agent_logger):
                await self.agent_logger(agent_logs)
            else:
                cast(Callable[[Dict[str, Any]], str], self.agent_logger)(agent_logs)

    async def on_status_logger(self, status: TestCaseStatus) -> None:
        if self.status_logger:
            if inspect.iscoroutinefunction(self.agent_logger):
                await self.status_logger(status)
            else:
                cast(Callable[[str], str], self.status_logger)(status)

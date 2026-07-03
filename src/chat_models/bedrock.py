"""Anthropic Bedrock client with flexible authentication support."""
import logging
from typing import Literal, Optional

import httpx
from anthropic import AsyncAnthropicBedrock
from anthropic.lib.bedrock._auth import get_auth_headers
from typing_extensions import override

logger = logging.getLogger(__name__)


class AnthropicBedrockClient(AsyncAnthropicBedrock):
    """AsyncAnthropicBedrock client with flexible authentication support.

    Supports both standard AWS SigV4 auth and custom header-based auth
    (e.g. for AI Hub or proxy endpoints).
    """

    def __init__(
        self,
        auth_mode: Literal["auto", "custom"] = "auto",
        custom_endpoint: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize client with flexible authentication.

        Args:
            auth_mode:
                - "auto": Standard AWS SigV4 authentication (default)
                - "custom": Header-based auth for AI Hub / proxy endpoints
            custom_endpoint: Base URL override (only used in "custom" mode)
        """
        self.auth_mode = auth_mode
        self.custom_endpoint = custom_endpoint

        if auth_mode == "custom" and custom_endpoint:
            kwargs["base_url"] = custom_endpoint

        super().__init__(**kwargs)

    @override
    async def _prepare_request(self, request: httpx.Request) -> None:
        if self.default_headers and self.auth_mode == "custom":
            for key, value in self.default_headers.items():
                request.headers[key] = f"{value}"
            request.headers["Content-Type"] = "application/json"
        else:
            data = request.read().decode()

            headers = get_auth_headers(
                method=request.method,
                url=str(request.url),
                headers=request.headers,
                aws_access_key=self.aws_access_key,
                aws_secret_key=self.aws_secret_key,
                aws_session_token=self.aws_session_token,
                region=self.aws_region or "us-east-1",
                profile=self.aws_profile,
                data=data,
            )
            request.headers.update(headers)

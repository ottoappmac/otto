"""ChatMLX wrapper that avoids streaming to work around tokenizers.Encoding bug.

langchain_community ChatMLX._stream assumes apply_chat_template(return_tensors="np")
returns a numpy array, but newer tokenizers return tokenizers.Encoding, causing:
    ValueError: Invalid type tokenizers.Encoding received in array initialization.

This wrapper overrides _astream to use _agenerate instead, bypassing the buggy _stream.
"""

from typing import Any, AsyncIterator, List, Optional

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_community.chat_models.mlx import ChatMLX


class ChatMLXNoStream(ChatMLX):
    """ChatMLX that avoids streaming to work around tokenizers.Encoding bug.

    When _astream is called, uses _agenerate and yields the full response as one chunk.
    This bypasses ChatMLX._stream which fails with newer tokenizers.
    """

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream by generating once and yielding the full result."""
        result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        text = result.generations[0].text
        if text:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(text, chunk=chunk)

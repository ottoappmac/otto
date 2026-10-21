"""oMLX replay keeps a prefix through the previous assistant turn.

The stand-in template encodes the Qwen rule that matters here: an assistant
turn before the latest real user message drops its think block unless
``preserve_thinking`` is set. Otto's stamp puts the reasoning back on the
request and sets that flag, so the next render starts with the previous one.
"""

from __future__ import annotations

from jinja2 import Environment

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deep_agent.model_factory import stamp_omlx_replay

_TEMPLATE = Environment().from_string(
    """
{%- set ns = namespace(last_query_index=-1) -%}
{%- for m in messages -%}
{%- if m.role == 'user' and not (m.content is string and m.content.startswith('<tool_response>')) -%}
{%- set ns.last_query_index = loop.index0 -%}
{%- endif -%}
{%- endfor -%}
{%- for m in messages -%}
<|im_start|>{{ m.role }}
{%- if m.role == 'assistant' and m.reasoning_content and (preserve_thinking or loop.index0 > ns.last_query_index) -%}
<think>
{{ m.reasoning_content }}
</think>
{%- endif -%}
{{ m.content }}<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- if enable_thinking is defined and not enable_thinking -%}
<think>

</think>

{%- endif -%}
{%- endif -%}
"""
)


def _render(messages: list[dict], *, preserve_thinking: bool, add_generation_prompt: bool) -> str:
    return _TEMPLATE.render(
        messages=messages,
        preserve_thinking=preserve_thinking,
        enable_thinking=False,
        add_generation_prompt=add_generation_prompt,
    )


def _payload(messages: list) -> dict:
    payload = {
        "messages": [
            {
                "role": "system" if isinstance(message, SystemMessage) else
                "user" if isinstance(message, HumanMessage) else "assistant",
                "content": message.content,
            }
            for message in messages
        ],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    stamp_omlx_replay(payload, messages)
    return payload


def test_stamp_sets_preserve_thinking_and_reasoning():
    messages = [
        SystemMessage(content="You are Otto."),
        HumanMessage(content="read the notes"),
        AIMessage(
            content="I'll read it.",
            additional_kwargs={"reasoning_content": "because the notes are the source"},
            tool_calls=[{
                "name": "read_file",
                "args": {"file_path": "notes.md"},
                "id": "call_1",
                "type": "tool_call",
            }],
        ),
    ]
    payload = _payload(messages)
    assert payload["extra_body"]["chat_template_kwargs"]["preserve_thinking"] is True
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assistant = payload["messages"][2]
    assert assistant["reasoning_content"] == "because the notes are the source"


def test_consecutive_renders_share_a_prefix_through_the_prior_assistant():
    history = [
        SystemMessage(content="You are Otto."),
        HumanMessage(content="read the notes"),
    ]
    first = _payload(history)
    first_history = _render(
        first["messages"],
        preserve_thinking=True,
        add_generation_prompt=False,
    )

    reply = AIMessage(
        content="I'll read it.",
        additional_kwargs={"reasoning_content": "because the notes are the source"},
    )
    second = _payload(history + [reply])
    kwargs = second["extra_body"]["chat_template_kwargs"]
    second_text = _render(
        second["messages"],
        preserve_thinking=kwargs["preserve_thinking"],
        add_generation_prompt=True,
    )
    assert second_text.startswith(first_history)
    assert "<think>\nbecause the notes are the source\n</think>" in second_text

    # A later user message moves the template's last-query index. The think
    # block stays only because the stamp set preserve_thinking.
    followed = second["messages"] + [{"role": "user", "content": "thanks"}]
    kept = _render(followed, preserve_thinking=True, add_generation_prompt=False)
    dropped = _render(followed, preserve_thinking=False, add_generation_prompt=False)
    through_assistant = _render(
        second["messages"],
        preserve_thinking=True,
        add_generation_prompt=False,
    )
    assert kept.startswith(through_assistant)
    assert not dropped.startswith(through_assistant)
    assert "because the notes are the source" not in dropped

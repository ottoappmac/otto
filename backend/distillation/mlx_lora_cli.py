"""mlx_lm.lora entry that detaches MoE routing indices from the VJP graph.

Qwen3-Next / Qwen3.6 SparseMoeBlock does:

    inds = argpartition(gate(x))
    scores = take_along_axis(gates, inds)

``x`` depends on trainable attention LoRA, so ``inds`` does too.  MLX
cannot VJP ``gather``, ``gather_axis``, or ``gather_mm`` w.r.t. indices.
The expert sort then indexes with ``argsort(inds)``, which is the
``[gather]`` failure.  Expert assignment is discrete — stop the index
gradient, keep the score and weight gradients.
"""

from __future__ import annotations

from typing import Any, Callable


def _wrap_rhs_indices(fn: Callable[..., Any]) -> Callable[..., Any]:
    import mlx.core as mx

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("rhs_indices") is not None:
            kwargs = {**kwargs, "rhs_indices": mx.stop_gradient(kwargs["rhs_indices"])}
        return fn(*args, **kwargs)

    return wrapped


def _wrap_stop_output(fn: Callable[..., Any]) -> Callable[..., Any]:
    import mlx.core as mx

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return mx.stop_gradient(fn(*args, **kwargs))

    return wrapped


def install_moe_index_stop_gradient() -> None:
    """Patch MLX so MoE routing indices do not request a VJP."""
    import mlx.core as mx

    if getattr(mx, "_otto_moe_index_stop", False):
        return

    _orig_take = mx.take_along_axis

    def _take(a, indices, *args, **kwargs):  # noqa: ANN001
        return _orig_take(a, mx.stop_gradient(indices), *args, **kwargs)

    mx.take_along_axis = _take
    # argpartition/argsort feed both take_along_axis and ``x[order]``
    # (the gather primitive inside expert sorting).
    mx.argpartition = _wrap_stop_output(mx.argpartition)
    mx.argsort = _wrap_stop_output(mx.argsort)
    for name in ("gather_mm", "gather_qmm", "gather_qqmm"):
        if hasattr(mx, name):
            setattr(mx, name, _wrap_rhs_indices(getattr(mx, name)))
    mx._otto_moe_index_stop = True  # type: ignore[attr-defined]


def main() -> None:
    install_moe_index_stop_gradient()
    from mlx_lm.lora import main as lora_main

    lora_main()


if __name__ == "__main__":
    main()

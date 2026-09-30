"""Offline LoRA train helpers — no GPU required."""

from __future__ import annotations

import json
from pathlib import Path

from backend.distillation.adapter_meta import read_adapter_meta
from backend.distillation.train import (
    build_lora_argv,
    looks_moe,
    prepare_lora_data,
    train_lora,
    window_messages,
    write_moe_lora_config,
)


def _row(*pairs: tuple[str, str]) -> str:
    messages = [{"role": role, "content": text} for role, text in pairs]
    return json.dumps({"messages": messages})


def test_prepare_lora_data_holds_out_shortest(tmp_path: Path):
    src = tmp_path / "sft.jsonl"
    short = _row(("user", "a"), ("assistant", "1"))
    mid = _row(("user", "bbbb"), ("assistant", "22"))
    long = _row(("user", "cccccc"), ("assistant", "333"))
    src.write_text(short + "\n" + mid + "\n" + long + "\n", encoding="utf-8")
    dest = tmp_path / "data"
    prepare_lora_data(src, dest)
    train = [
        json.loads(ln)
        for ln in (dest / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    valid = json.loads((dest / "valid.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert valid["messages"][-1]["content"] == "1"
    contents = {r["messages"][-1]["content"] for r in train}
    assert contents == {"22", "333"}


def test_prepare_single_line_duplicates_valid(tmp_path: Path):
    src = tmp_path / "sft.jsonl"
    line = _row(("user", "hi"), ("assistant", "yo"))
    src.write_text(line + "\n", encoding="utf-8")
    dest = tmp_path / "data"
    prepare_lora_data(src, dest)
    train = json.loads((dest / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    valid = json.loads((dest / "valid.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert train["messages"][-1]["content"] == "yo"
    assert valid == train


def test_build_lora_argv_contains_mask_prompt_and_student():
    argv = build_lora_argv(
        student_id="mlx-community/Qwen3-8B-4bit",
        data_dir=Path("/tmp/data"),
        adapter_path=Path("/tmp/ad"),
        iters=50,
    )
    assert argv[argv.index("-m") + 1] == "mlx_lm"
    assert "lora" in argv
    assert "--train" in argv
    assert "--mask-prompt" in argv
    assert argv[argv.index("--model") + 1] == "mlx-community/Qwen3-8B-4bit"
    assert argv[argv.index("--iters") + 1] == "50"
    assert argv[argv.index("--num-layers") + 1] == "16"
    assert argv[argv.index("--max-seq-length") + 1] == "2048"
    assert "--grad-checkpoint" in argv


def test_build_lora_argv_moe_skips_grad_checkpoint_and_uses_config(tmp_path: Path):
    cfg = tmp_path / "lora.yaml"
    cfg.write_text("fine_tune_type: lora\n", encoding="utf-8")
    argv = build_lora_argv(
        student_id="mlx-community/Qwen3.6-35B-A3B-4bit",
        data_dir=tmp_path,
        adapter_path=tmp_path / "ad",
        config_path=cfg,
    )
    assert "--grad-checkpoint" in argv
    assert argv[argv.index("--config") + 1] == str(cfg)
    assert any(str(p).endswith("mlx_lora_cli.py") for p in argv)
    assert looks_moe("mlx-community/Qwen3.6-35B-A3B-4bit")
    assert not looks_moe("mlx-community/Qwen3-8B-4bit")


def test_moe_lora_config_skips_router_and_switch_experts(tmp_path: Path):
    path = write_moe_lora_config(tmp_path / "lora.yaml")
    text = path.read_text(encoding="utf-8")
    assert "- mlp.gate_proj" in text
    assert "- mlp.shared_expert.gate_proj" in text
    assert "- self_attn.q_proj" in text
    assert "switch_mlp" not in text
    assert "linear_attn" not in text
    assert not any(line.strip() == "- mlp.gate" for line in text.splitlines())


def test_window_messages_keeps_trailing_turn():
    messages = [
        {"role": "user", "content": "x" * 9000},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "short question"},
        {"role": "assistant", "content": "short answer"},
    ]
    window = window_messages(messages, max_tokens=200)
    roles = [m["role"] for m in window]
    assert roles[0] == "user"
    assert roles[-1] == "assistant"
    assert window[-1]["content"] == "short answer"
    assert "x" * 20 not in "".join(str(m.get("content")) for m in window)


def test_window_messages_crops_huge_last_user():
    messages = [
        {"role": "user", "content": "y" * 20000},
        {"role": "assistant", "content": "keep-this-reply"},
    ]
    window = window_messages(messages, max_tokens=200)
    assert window[-1]["content"] == "keep-this-reply"
    assert window[0]["role"] == "user"
    assert len(window[0]["content"]) < 20000


def test_prepare_lora_data_supervises_tool_call_turns(tmp_path: Path):
    src = tmp_path / "sft.jsonl"
    obj = {
        "messages": [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "web_research", "arguments": {"q": "x"}},
                }],
            },
            {"role": "tool", "content": "result", "tool_call_id": "c1", "name": "web_research"},
            {"role": "assistant", "content": "Here is the written summary."},
        ]
    }
    src.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    dest = tmp_path / "data"
    prepare_lora_data(src, dest, max_seq_length=2048)
    rows = []
    for name in ("train.jsonl", "valid.jsonl"):
        rows.extend(
            json.loads(ln)
            for ln in (dest / name).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
    endings = [r["messages"][-1] for r in rows]
    assert any(m.get("tool_calls") for m in endings)
    assert any(m.get("content") == "Here is the written summary." for m in endings)


def test_prepare_lora_data_windows_and_holds_out_shortest(tmp_path: Path):
    src = tmp_path / "sft.jsonl"
    long_msgs = {
        "messages": [
            {"role": "user", "content": "a" * 8000},
            {"role": "assistant", "content": "b" * 8000},
            {"role": "user", "content": "keep me"},
            {"role": "assistant", "content": "and me"},
        ]
    }
    short = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    mid = {"messages": [{"role": "user", "content": "mid"}, {"role": "assistant", "content": "ok"}]}
    src.write_text(
        json.dumps(long_msgs) + "\n" + json.dumps(mid) + "\n" + json.dumps(short) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "data"
    prepare_lora_data(src, dest, max_seq_length=400)
    valid = json.loads((dest / "valid.jsonl").read_text(encoding="utf-8").splitlines()[0])
    train = [
        json.loads(ln)
        for ln in (dest / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert valid["messages"][-1]["content"] in {"yo", "ok", "and me"}
    long_row = next(r for r in train + [valid] if any((m.get("content") == "and me") for m in r["messages"]))
    assert all("a" * 100 not in str(m.get("content")) for m in long_row["messages"])


def test_parse_iter_from_lora_log():
    from backend.distillation.train import parse_iter_from_line

    assert parse_iter_from_line("Iter 12: Train loss 1.23, It/sec 0.4") == 12
    assert parse_iter_from_line("Iteration 3: Train loss 2.0") == 3
    assert parse_iter_from_line("Saved adapter weights") is None


def test_train_lora_dry_run_writes_meta(tmp_path: Path, monkeypatch):
    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        json.dumps({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "backend.distillation.train.adapters_dir", lambda **k: tmp_path / "adapters",
    )
    monkeypatch.setattr(
        "backend.distillation.train.runs_dir", lambda **k: tmp_path / "runs",
    )
    out = tmp_path / "adapters" / "activity"
    result = train_lora(
        sft_jsonl=sft,
        student_id="mlx-community/Qwen3-8B-4bit",
        teacher_id="mlx-community/Qwen3-32B-4bit",
        adapter_path=out,
        run=False,
        allow_family_mismatch=True,
        purpose="mail triage from activity traces",
    )
    assert result["exit_code"] is None
    meta = read_adapter_meta(out)
    assert meta["base_repo_id"] == "mlx-community/Qwen3-8B-4bit"
    assert meta["teacher_model_id"] == "mlx-community/Qwen3-32B-4bit"
    assert meta["purpose"] == "mail triage from activity traces"
    assert result["purpose"] == "mail triage from activity traces"
    assert (tmp_path / "runs" / "activity" / "train.jsonl").is_file()
    assert result["catalog_id"] == "otto-distill/activity"
    assert "distilled" in result["display_name"]
    assert "--config" not in result["argv"]
    assert not (tmp_path / "runs" / "activity" / "lora.yaml").is_file()


def test_train_lora_moe_student_writes_router_safe_config(tmp_path: Path, monkeypatch):
    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        _row(("user", "hi"), ("assistant", "hello")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "backend.distillation.train.adapters_dir", lambda **k: tmp_path / "adapters",
    )
    monkeypatch.setattr(
        "backend.distillation.train.runs_dir", lambda **k: tmp_path / "runs",
    )
    out = tmp_path / "adapters" / "activity"
    result = train_lora(
        sft_jsonl=sft,
        student_id="mlx-community/Qwen3.6-35B-A3B-4bit",
        teacher_id="mlx-community/Qwen3.6-35B-A3B-8bit",
        adapter_path=out,
        run=False,
        allow_family_mismatch=True,
    )
    cfg = tmp_path / "runs" / "activity" / "lora.yaml"
    assert cfg.is_file()
    assert "--config" in result["argv"]
    assert any(str(p).endswith("mlx_lora_cli.py") for p in result["argv"])
    assert any("Mixture-of-Experts" in w for w in result["warnings"])
    text = cfg.read_text(encoding="utf-8")
    assert "switch_mlp" not in text
    assert "linear_attn" not in text
    assert argv_has_max_seq(result["argv"], "1024")
    assert not any(line.strip() == "- mlp.gate" for line in text.splitlines())


def argv_has_max_seq(argv: list[str], value: str) -> bool:
    return argv[argv.index("--max-seq-length") + 1] == value


def test_train_lora_hybrid_freezes_linear_attention(tmp_path: Path, monkeypatch):
    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        _row(("user", "hi"), ("assistant", "hello")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "backend.distillation.train.adapters_dir", lambda **k: tmp_path / "adapters",
    )
    monkeypatch.setattr(
        "backend.distillation.train.runs_dir", lambda **k: tmp_path / "runs",
    )
    out = tmp_path / "adapters" / "activity"
    result = train_lora(
        sft_jsonl=sft,
        student_id="mlx-community/Qwen3.5-9B-4bit",
        teacher_id="mlx-community/Qwen3.6-35B-A3B-8bit",
        adapter_path=out,
        run=False,
        allow_family_mismatch=True,
    )
    text = (tmp_path / "runs" / "activity" / "lora.yaml").read_text(encoding="utf-8")
    assert "linear_attn" not in text
    assert "- self_attn.q_proj" in text
    assert "--config" in result["argv"]
    assert not any(str(p).endswith("mlx_lora_cli.py") for p in result["argv"])
    assert argv_has_max_seq(result["argv"], "2048")
    assert any("Gated DeltaNet" in w for w in result["warnings"])


def test_moe_cli_detaches_gather_indices():
    from backend.distillation.mlx_lora_cli import install_moe_index_stop_gradient

    import mlx.core as mx

    install_moe_index_stop_gradient()

    def loss(x):
        gates = mx.softmax(x, axis=-1)
        inds = mx.argpartition(gates, kth=-2, axis=-1)[..., -2:]
        order = mx.argsort(inds.reshape(-1))
        gathered = x.reshape(-1)[order]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        return gathered.sum() + scores.sum()

    val, grad = mx.value_and_grad(loss)(mx.random.normal((4, 8)))
    mx.eval(val, grad)
    assert getattr(mx, "_otto_moe_index_stop", False)

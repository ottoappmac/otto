"""Yahoo alternate-layout parsing for the ddgs-backed web search."""

import pytest

pytest.importorskip("ddgs")

from ddgs.engines.yahoo import Yahoo  # noqa: E402

from tools.research import _loaders  # noqa: E402

# Trimmed from a real Yahoo response that ddgs's stock parser returns
# zero results for (no ``relsrch`` blocks; results are ``section.algo-sr``).
ALT_LAYOUT_HTML = """
<html><body><section id="main-algo"><section id="algo-bd">
<section class="reg searchCenterMiddle">
  <section class="dd fst algo s-algo algo-sr Sr">
    <div class="compTitle p-r"><h3 class="title d-ib mt-42">
      <a class="s-title fz-m" aria-label="BenchCAD &mdash; a benchmark for programmatic CAD"
         href="https://r.search.yahoo.com/_ylt=x;_ylu=y/RV=2/RE=1/RO=10/RU=https%3a%2f%2fbenchcad.com%2f/RK=2/RS=z-">
        <span class="title-url"><span>benchcad.com</span>https://<b>benchcad</b>.com</span>BenchCAD &mdash; a benchmark</a>
    </h3></div>
    <div class="compText"><p>17,900 execution-verified CadQuery programs.</p></div>
  </section>
  <section class="dd algo s-algo algo-sr Sr">
    <div class="compTitle p-r"><h3>
      <a aria-label="BenchCAD paper" href="https://r.search.yahoo.com/_ylt=a/RU=https%3a%2f%2farxiv.org%2fabs%2f2605.10865/RK=2/RS=b-">x</a>
    </h3></div>
    <div class="compText">We present BenchCAD.</div>
  </section>
</section></section></section></body></html>
"""


def test_stock_parser_misses_alt_layout():
    # Guards the premise: if ddgs ever parses this itself, the patch is moot.
    _loaders._yahoo_patched = False
    import importlib

    import ddgs.engines.yahoo as yahoo_mod

    importlib.reload(yahoo_mod)
    assert yahoo_mod.Yahoo().extract_results(ALT_LAYOUT_HTML) == []


def test_patched_parser_extracts_alt_layout():
    _loaders._yahoo_patched = False
    import importlib

    import ddgs.engines.yahoo as yahoo_mod

    importlib.reload(yahoo_mod)
    _loaders._patch_yahoo_alt_layout()
    engine = yahoo_mod.Yahoo()
    results = engine.post_extract_results(engine.extract_results(ALT_LAYOUT_HTML))
    assert [r.href for r in results] == ["https://benchcad.com/", "https://arxiv.org/abs/2605.10865"]
    assert results[0].title == "BenchCAD — a benchmark for programmatic CAD"
    assert "CadQuery" in results[0].body


def test_patch_is_idempotent():
    _loaders._patch_yahoo_alt_layout()
    first = Yahoo.extract_results
    _loaders._patch_yahoo_alt_layout()
    assert Yahoo.extract_results is first

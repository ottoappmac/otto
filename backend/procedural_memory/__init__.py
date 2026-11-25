"""Procedural memory: what worked and what failed when Otto's agents drove macOS and the browser.

Design record: ``docs/procedural-memory-plan.md``. In short:

* Every delegated desktop / browser agent run is saved as an **episode** record when it ends (``ingest``), with
  ``status='pending'`` and no vectors, so nothing unlabelled can be retrieved.
* A trusted label (benchmark score, user feedback, a calibrated judge) **admits** it (``service.label_run``):
  only then are its retrievable pieces written and embedded. Failed runs contribute the segments the trace itself
  confirmed, deterministic no-op facts and lessons phrased as checks; never their whole step sequence.
* ``recall_procedure`` (``tool``) returns one item (recipe + facts + lessons) above a similarity floor, under a token
  cap, from a store that is read-only in benchmark runs (a frozen snapshot).

Modules: ``compact`` (transcript -> episode, deterministic), ``store`` (SQLite), ``embed``, ``policy`` (labels and
admission), ``enrich`` (LLM intents and lessons, with a deterministic fallback), ``retrieve`` (hybrid search and
rendering), ``service`` (paths, modes, the entry points the app and the benchmark call), ``tool``.
"""

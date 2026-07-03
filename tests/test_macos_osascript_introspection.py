"""Unit tests for the macos-osascript MCP's introspection helpers.

The two new tools — ``inspect_app_dictionary`` and ``dump_ax_tree`` —
exist to replace LLM-side priors with deterministic system probes.

The decorated tool entry points run in their own uv-provisioned venv
at runtime and can't be cleanly imported from the parent process's
test environment (FastMCP parameter introspection differs across
versions).  All testable logic is therefore extracted into the
``_helpers`` module which is pure-Python and free of MCP decoration.

This file covers:

* The ``sdef`` classifier correctly distinguishes dictionary-rich apps
  (Mail-style XML) from dictionary-empty ones (Slack / Electron-style
  standard-suite-only XML).
* App-bundle resolution searches the conventional install locations.
* AppleScript string escaping handles embedded quotes / backslashes.

End-to-end macOS-only behaviour (real ``sdef`` output, real System
Events queries) lives in the macOS integration test matrix.
"""

from __future__ import annotations

from pathlib import Path


# ── Sdef classifier ---------------------------------------------------------


def test_classify_dictionary_detects_dictionary_rich_app():
    """Mail-shaped sdef: real Suite, app-specific class, several verbs."""
    from backend.builtin_mcps.macos_osascript._helpers import classify_dictionary

    sdef_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<dictionary title="Mail Terminology">
  <suite name="Mail" code="emal">
    <class name="message" code="bcke" />
    <class name="mailbox" code="mbxr" />
    <class name="account" code="mAcc" />
    <command name="send" code="emalsend" />
    <command name="check for new mail" code="emalcfnm" />
  </suite>
</dictionary>
"""
    has_specific, classes, commands, _props, _elems = classify_dictionary(sdef_xml)
    assert has_specific is True
    assert "message" in classes
    assert "mailbox" in classes
    assert "send" in commands
    assert "check for new mail" in commands


def test_classify_dictionary_detects_electron_app_as_empty():
    """Slack-shaped sdef: only standard suite (NSCoreSuite-style classes).

    The classifier must report ``has_app_specific_suite=False`` so the
    agent skips the dictionary path entirely and goes straight to the
    AX-tree path — exactly what the failing Slack DM trace needed.
    """
    from backend.builtin_mcps.macos_osascript._helpers import classify_dictionary

    sdef_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<dictionary title="Slack Terminology">
  <suite name="Standard Suite" code="????">
    <class name="application" code="capp" />
    <class name="document" code="docu" />
    <class name="window" code="cwin" />
  </suite>
</dictionary>
"""
    has_specific, classes, commands, _props, _elems = classify_dictionary(sdef_xml)
    assert has_specific is False
    assert classes == []
    assert commands == []


def test_classify_dictionary_handles_no_suite():
    """Pathological sdef with no <class>/<command> entries at all."""
    from backend.builtin_mcps.macos_osascript._helpers import classify_dictionary

    has_specific, classes, commands, properties, elements = classify_dictionary(
        "<dictionary/>"
    )
    assert has_specific is False
    assert classes == []
    assert commands == []
    assert properties == {}
    assert elements == {}


def test_classify_dictionary_strips_standard_classes_only():
    """A real-world sdef may declare standard classes alongside
    app-specific ones.  The classifier strips standard names but keeps
    app-specific ones."""
    from backend.builtin_mcps.macos_osascript._helpers import classify_dictionary

    sdef_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<dictionary>
  <suite>
    <class name="application" code="capp" />
    <class name="document" code="docu" />
    <class name="window" code="cwin" />
    <class name="reminder" code="remi" />
    <class name="reminder list" code="reli" />
  </suite>
</dictionary>
"""
    has_specific, classes, commands, _props, _elems = classify_dictionary(sdef_xml)
    assert has_specific is True
    assert "reminder" in classes
    assert "reminder list" in classes
    # Standard-class names must be stripped from the result.
    assert "application" not in classes
    assert "window" not in classes


def test_classify_dictionary_recognizes_only_commands():
    """Some apps expose verbs but no app-specific classes — still counts
    as a real dictionary because there's something to call."""
    from backend.builtin_mcps.macos_osascript._helpers import classify_dictionary

    sdef_xml = """\
<dictionary>
  <suite>
    <command name="custom verb" code="abcd1234" />
  </suite>
</dictionary>
"""
    has_specific, classes, commands, _props, _elems = classify_dictionary(sdef_xml)
    assert has_specific is True
    assert "custom verb" in commands


def test_classify_dictionary_extracts_properties_from_class_and_extension():
    """Outlook-shaped sdef: most ``message`` fields live on a separate
    ``<class-extension extends="message">`` block, and ``sender`` is its
    own class (so ``sender of msg`` is a property/class collision — the
    exact ``-2741 "found class name"`` trap).  The classifier must merge
    properties from both the ``<class>`` body and the extension so the
    agent reads real field names instead of guessing."""
    from backend.builtin_mcps.macos_osascript._helpers import classify_dictionary

    sdef_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<dictionary title="Microsoft Outlook Terminology">
  <suite name="Microsoft Outlook Suite" code="OUTL">
    <class name="mailbox" code="mbxr" inherits="folder">
      <element type="message" />
    </class>
    <class name="message" code="cMSG" inherits="item">
      <property name="subject" code="subj" type="text" />
    </class>
    <class-extension extends="message">
      <property name="time received" code="tRCV" type="date" />
      <property name="time sent" code="tSNT" type="date" />
      <property name="is read" code="isRD" type="boolean" />
    </class-extension>
    <class name="sender" code="sndr" inherits="item">
      <property name="address" code="addr" type="text" />
    </class>
  </suite>
</dictionary>
"""
    has_specific, classes, commands, properties, elements = classify_dictionary(
        sdef_xml
    )
    assert has_specific is True
    assert "message" in classes
    assert "sender" in classes
    # Properties merged from both <class> and <class-extension>.
    assert properties["message"] == [
        "is read", "subject", "time received", "time sent",
    ]
    assert properties["sender"] == ["address"]
    # The Mail idiom the agent kept guessing is absent — proving the probe
    # would steer it away from `date received` / `sender of msg`.
    assert "date received" not in properties["message"]
    # Containment: mailbox holds messages, so the agent knows to write
    # `messages of mailbox`, not the `items of` fallback that the failing
    # Outlook session resorted to.
    assert elements["mailbox"] == ["message"]


# ── App-path resolver -------------------------------------------------------


def test_resolve_app_path_finds_existing_bundle(tmp_path, monkeypatch):
    """Stub Path.home() to point at tmp_path and seed a fake bundle so
    the test is hermetic and host-independent."""
    from backend.builtin_mcps.macos_osascript import _helpers

    fake_app = tmp_path / "Applications" / "FakeApp.app"
    fake_app.mkdir(parents=True)

    monkeypatch.setattr(_helpers.Path, "home", lambda: tmp_path)

    resolved = _helpers.resolve_app_path("FakeApp")
    # On a CI host with no /Applications/FakeApp.app, the matching
    # candidate is the fake ``~/Applications/FakeApp.app`` we seeded.
    assert resolved == str(fake_app)


def test_resolve_app_path_returns_none_for_missing(monkeypatch):
    from backend.builtin_mcps.macos_osascript import _helpers

    monkeypatch.setattr(
        _helpers.Path, "home",
        lambda: Path("/tmp/__never_exists_for_test__"),
    )
    assert _helpers.resolve_app_path("__definitely_not_installed__") is None


# ── AppleScript string escaping ---------------------------------------------


def test_escape_applescript_string_handles_double_quotes():
    """Slack and other apps may have quotes in window titles.  Our
    dump_ax_tree script embeds the app name in an AppleScript string
    literal, so unescaped quotes would terminate the literal early and
    produce the same -2741 syntax errors the agent should be fixing."""
    from backend.builtin_mcps.macos_osascript._helpers import escape_applescript_string

    assert escape_applescript_string('Hello "world"') == 'Hello \\"world\\"'


def test_escape_applescript_string_handles_backslashes():
    from backend.builtin_mcps.macos_osascript._helpers import escape_applescript_string

    # Backslash-double-quote: backslash doubled, then quote escaped.
    assert escape_applescript_string('a\\b"c') == 'a\\\\b\\"c'


def test_escape_applescript_string_passthrough_for_plain_text():
    from backend.builtin_mcps.macos_osascript._helpers import escape_applescript_string

    assert escape_applescript_string("Slack") == "Slack"
    assert escape_applescript_string("VS Code") == "VS Code"


# ── Desktop-lease classification --------------------------------------------
#
# Only focus-stealing / input-synthesizing scripts must take the desktop
# lease.  A false negative here means two agents collide on the
# foreground, so these tests pin the conservative behavior.


def test_script_needs_desktop_flags_activate():
    from backend.builtin_mcps.macos_osascript._helpers import script_needs_desktop

    assert script_needs_desktop('tell application "Slack" to activate') is True


def test_script_needs_desktop_flags_synthesized_input():
    from backend.builtin_mcps.macos_osascript._helpers import script_needs_desktop

    assert script_needs_desktop(
        'tell application "System Events" to keystroke "hello"'
    ) is True
    assert script_needs_desktop(
        'tell application "System Events" to key code 36'
    ) is True
    assert script_needs_desktop(
        'click button "Send" of window 1 of process "Slack"'
    ) is True


def test_script_needs_desktop_is_case_insensitive():
    from backend.builtin_mcps.macos_osascript._helpers import script_needs_desktop

    assert script_needs_desktop('tell application "Mail" to ACTIVATE') is True


def test_script_needs_desktop_flags_jxa_activate():
    from backend.builtin_mcps.macos_osascript._helpers import script_needs_desktop

    # JXA: Application("Slack").activate() still contains the "activate" token.
    assert script_needs_desktop('Application("Slack").activate()') is True


def test_script_needs_desktop_allows_background_dictionary_call():
    from backend.builtin_mcps.macos_osascript._helpers import script_needs_desktop

    # Pure dictionary read — runs in the background, no focus needed.
    assert script_needs_desktop(
        'tell application "Mail" to get subject of every message of inbox'
    ) is False


def test_script_needs_desktop_allows_readonly_ax_query():
    from backend.builtin_mcps.macos_osascript._helpers import script_needs_desktop

    # Reading AX contents without activating does not steal focus.
    assert script_needs_desktop(
        'tell application "System Events" to get entire contents of window 1 '
        'of process "Notes"'
    ) is False


def test_script_needs_desktop_handles_empty():
    from backend.builtin_mcps.macos_osascript._helpers import script_needs_desktop

    assert script_needs_desktop("") is False


# ── Virtual POSIX-path remap ------------------------------------------------
#
# Inline AppleScript runs on the host, so a ``POSIX file "/output/foo"``
# resolves against the host root rather than the session sandbox — the bug
# behind "email sent without the attachment".  These tests pin the remap so
# only genuine ``POSIX file`` literals are touched and real host paths /
# prose are left intact.


def test_rewrite_remaps_virtual_output_path(tmp_path):
    """A POSIX file pointing at /output/<file> that exists under the sandbox
    is rewritten to the real host path."""
    from backend.builtin_mcps.macos_osascript._helpers import (
        rewrite_virtual_posix_paths,
    )

    out = tmp_path / "output"
    out.mkdir()
    (out / "report.pdf").write_text("x")

    script = 'make new attachment with properties {file name:(POSIX file "/output/report.pdf")}'
    rewritten, rewrites = rewrite_virtual_posix_paths(
        script, session_files=str(tmp_path)
    )

    real = str(out / "report.pdf")
    assert f'POSIX file "{real}"' in rewritten
    # The root-anchored literal is gone (the new path legitimately *ends*
    # with /output/report.pdf, so check the quoted literal, not a substring).
    assert 'POSIX file "/output/report.pdf"' not in rewritten
    assert rewrites == [("/output/report.pdf", real)]


def test_rewrite_remaps_when_only_parent_exists(tmp_path):
    """A file about to be created (parent dir exists, file does not) is still
    remapped, so a script that writes into /output works too."""
    from backend.builtin_mcps.macos_osascript._helpers import (
        rewrite_virtual_posix_paths,
    )

    (tmp_path / "output").mkdir()

    script = 'set f to open for access (POSIX file "/output/new.txt") with write permission'
    rewritten, rewrites = rewrite_virtual_posix_paths(
        script, session_files=str(tmp_path)
    )

    assert str(tmp_path / "output" / "new.txt") in rewritten
    assert len(rewrites) == 1


def test_rewrite_leaves_real_host_paths_untouched(tmp_path):
    """An existing host path (the tmp file itself) must not be remapped."""
    from backend.builtin_mcps.macos_osascript._helpers import (
        rewrite_virtual_posix_paths,
    )

    real_file = tmp_path / "real.txt"
    real_file.write_text("x")

    script = f'read (POSIX file "{real_file}")'
    rewritten, rewrites = rewrite_virtual_posix_paths(
        script, session_files=str(tmp_path)
    )

    assert rewritten == script
    assert rewrites == []


def test_rewrite_does_not_touch_free_text_paths(tmp_path):
    """Critically: a /output/ path in an email body (set content / subject)
    is NOT a POSIX file literal, so it must be left verbatim — otherwise the
    human-readable body gets mangled into a sandbox path."""
    from backend.builtin_mcps.macos_osascript._helpers import (
        rewrite_virtual_posix_paths,
    )

    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "report.pdf").write_text("x")

    script = (
        'set content to "Full report saved at /output/report.pdf"\n'
        'make new attachment with properties {file name:(POSIX file "/output/report.pdf")}'
    )
    rewritten, rewrites = rewrite_virtual_posix_paths(
        script, session_files=str(tmp_path)
    )

    # The prose mention is untouched ...
    assert "/output/report.pdf" in rewritten
    # ... while the POSIX file literal is remapped exactly once.
    assert rewrites == [
        ("/output/report.pdf", str(tmp_path / "output" / "report.pdf"))
    ]
    assert str(tmp_path / "output" / "report.pdf") in rewritten


def test_rewrite_skips_already_expanded_sandbox_paths(tmp_path):
    """A path already under $SESSION_FILES must not be double-prefixed."""
    from backend.builtin_mcps.macos_osascript._helpers import (
        rewrite_virtual_posix_paths,
    )

    out = tmp_path / "output"
    out.mkdir()
    (out / "report.pdf").write_text("x")
    already = str(out / "report.pdf")

    script = f'make new attachment with properties {{file name:(POSIX file "{already}")}}'
    rewritten, rewrites = rewrite_virtual_posix_paths(
        script, session_files=str(tmp_path)
    )

    assert rewritten == script
    assert rewrites == []


def test_rewrite_noop_without_session_files(tmp_path):
    """No SESSION_FILES configured → remap is disabled, script unchanged."""
    from backend.builtin_mcps.macos_osascript._helpers import (
        rewrite_virtual_posix_paths,
    )

    script = 'make new attachment with properties {file name:(POSIX file "/output/report.pdf")}'
    rewritten, rewrites = rewrite_virtual_posix_paths(script, session_files="")

    assert rewritten == script
    assert rewrites == []


def test_rewrite_handles_multiple_literals(tmp_path):
    """Several POSIX file literals in one script are each remapped."""
    from backend.builtin_mcps.macos_osascript._helpers import (
        rewrite_virtual_posix_paths,
    )

    out = tmp_path / "output"
    out.mkdir()
    for name in ("a.html", "b.html"):
        (out / name).write_text("x")

    script = (
        'make new attachment with properties {file name:(POSIX file "/output/a.html")}\n'
        'make new attachment with properties {file name:(POSIX file "/output/b.html")}'
    )
    rewritten, rewrites = rewrite_virtual_posix_paths(
        script, session_files=str(tmp_path)
    )

    assert len(rewrites) == 2
    assert 'POSIX file "/output/a.html"' not in rewritten
    assert 'POSIX file "/output/b.html"' not in rewritten
    assert f'POSIX file "{out / "a.html"}"' in rewritten
    assert f'POSIX file "{out / "b.html"}"' in rewritten

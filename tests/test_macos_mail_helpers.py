"""Unit tests for the macos-mail MCP's pure-Python helpers.

The decorated tool entry points in ``server.py`` run in their own
uv-provisioned venv at runtime and aren't imported here — see
``tests/test_macos_osascript_introspection.py`` for the same rationale.
All testable logic lives in ``_helpers``, which is pure Python and free
of MCP decoration.
"""

from __future__ import annotations


# ── AppleScript string escaping ---------------------------------------------


def test_escape_applescript_string_handles_double_quotes():
    from backend.builtin_mcps.macos_mail._helpers import escape_applescript_string

    assert escape_applescript_string('Hello "world"') == 'Hello \\"world\\"'


def test_escape_applescript_string_handles_backslashes():
    from backend.builtin_mcps.macos_mail._helpers import escape_applescript_string

    assert escape_applescript_string('a\\b"c') == 'a\\\\b\\"c'


def test_escape_applescript_string_passthrough_for_plain_text():
    from backend.builtin_mcps.macos_mail._helpers import escape_applescript_string

    assert escape_applescript_string("Weekly report") == "Weekly report"


def test_escape_applescript_string_handles_none_and_empty():
    from backend.builtin_mcps.macos_mail._helpers import escape_applescript_string

    assert escape_applescript_string("") == ""
    assert escape_applescript_string(None) == ""


# ── Mailbox reference resolution --------------------------------------------


def test_mailbox_reference_account_scoped():
    from backend.builtin_mcps.macos_mail._helpers import mailbox_reference

    assert (
        mailbox_reference("Archive", "Work") ==
        'mailbox "Archive" of account (my resolveAccountName("Work"))'
    )


def test_mailbox_reference_account_scoped_escapes_quotes():
    from backend.builtin_mcps.macos_mail._helpers import mailbox_reference

    ref = mailbox_reference('My "Special" Folder', 'Bob\'s Account')
    assert ref == (
        'mailbox "My \\"Special\\" Folder" '
        'of account (my resolveAccountName("Bob\'s Account"))'
    )


def test_mailbox_reference_alias_inbox_case_insensitive():
    from backend.builtin_mcps.macos_mail._helpers import mailbox_reference

    assert mailbox_reference("INBOX") == "inbox"
    assert mailbox_reference("Inbox") == "inbox"


def test_mailbox_reference_aliases_cover_special_mailboxes():
    from backend.builtin_mcps.macos_mail._helpers import mailbox_reference

    assert mailbox_reference("sent") == "sent mailbox"
    assert mailbox_reference("drafts") == "drafts mailbox"
    assert mailbox_reference("trash") == "trash mailbox"
    assert mailbox_reference("deleted") == "trash mailbox"
    assert mailbox_reference("junk") == "junk mailbox"
    assert mailbox_reference("spam") == "junk mailbox"
    assert mailbox_reference("outbox") == "outbox"


def test_mailbox_reference_aliases_do_not_apply_when_account_given():
    from backend.builtin_mcps.macos_mail._helpers import mailbox_reference

    # "inbox" must resolve to the account's own mailbox named "inbox",
    # not the app-level alias, once an account is specified.
    assert mailbox_reference("inbox", "Work") == (
        'mailbox "inbox" of account (my resolveAccountName("Work"))'
    )


def test_mailbox_reference_falls_back_to_app_wide_search_by_name():
    from backend.builtin_mcps.macos_mail._helpers import mailbox_reference

    assert mailbox_reference("Newsletters") == 'mailbox "Newsletters"'


# ── Message id-form reference ------------------------------------------------


def test_message_reference_shape():
    from backend.builtin_mcps.macos_mail._helpers import message_reference

    assert message_reference(12345, "inbox") == "(first message of inbox whose id is 12345)"


def test_message_reference_coerces_string_ids():
    from backend.builtin_mcps.macos_mail._helpers import message_reference

    assert message_reference("987", 'mailbox "INBOX" of account "Work"') == (
        '(first message of mailbox "INBOX" of account "Work" whose id is 987)'
    )


# ── Keyword whose-clause -----------------------------------------------------


def test_keyword_whose_clause_defaults_to_subject_and_sender_only():
    from backend.builtin_mcps.macos_mail._helpers import keyword_whose_clause

    clause = keyword_whose_clause("invoice")
    assert 'subject contains "invoice"' in clause
    assert 'sender contains "invoice"' in clause
    assert "content contains" not in clause


def test_keyword_whose_clause_include_body_adds_content_match():
    from backend.builtin_mcps.macos_mail._helpers import keyword_whose_clause

    clause = keyword_whose_clause("invoice", include_body=True)
    assert 'subject contains "invoice"' in clause
    assert 'sender contains "invoice"' in clause
    assert 'content contains "invoice"' in clause


def test_keyword_whose_clause_escapes_quotes():
    from backend.builtin_mcps.macos_mail._helpers import keyword_whose_clause

    clause = keyword_whose_clause('say "hi"')
    assert 'say \\"hi\\"' in clause


# ── Date literal for days_back -----------------------------------------------


def test_applescript_date_literal_zero_is_unbounded():
    from backend.builtin_mcps.macos_mail._helpers import applescript_date_literal

    assert applescript_date_literal(0) == ""
    assert applescript_date_literal(-5) == ""


def test_applescript_date_literal_positive_builds_expression():
    from backend.builtin_mcps.macos_mail._helpers import applescript_date_literal

    literal = applescript_date_literal(7)
    assert literal == "((current date) - (7 * days))"


# ── clamp_limit ---------------------------------------------------------------


def test_clamp_limit_bounds_low_and_high():
    from backend.builtin_mcps.macos_mail._helpers import clamp_limit

    assert clamp_limit(0) == 1
    assert clamp_limit(-10) == 1
    assert clamp_limit(50) == 50
    assert clamp_limit(10_000) == 200
    assert clamp_limit(10_000, cap=500) == 500


def test_clamp_limit_handles_non_numeric():
    from backend.builtin_mcps.macos_mail._helpers import clamp_limit

    assert clamp_limit(None) == 1
    assert clamp_limit("not a number") == 1


# ── Attachment path resolution ------------------------------------------------


def test_resolve_attachment_path_real_host_path(tmp_path):
    from backend.builtin_mcps.macos_mail._helpers import resolve_attachment_path

    real_file = tmp_path / "report.pdf"
    real_file.write_text("x")

    resolved = resolve_attachment_path(str(real_file))
    assert resolved == real_file


def test_resolve_attachment_path_remaps_virtual_session_path(tmp_path):
    from backend.builtin_mcps.macos_mail._helpers import resolve_attachment_path

    out = tmp_path / "output"
    out.mkdir()
    (out / "report.pdf").write_text("x")

    resolved = resolve_attachment_path("/output/report.pdf", session_files=str(tmp_path))
    assert resolved == out / "report.pdf"


def test_resolve_attachment_path_missing_returns_none(tmp_path):
    from backend.builtin_mcps.macos_mail._helpers import resolve_attachment_path

    assert resolve_attachment_path("/output/nope.pdf", session_files=str(tmp_path)) is None


def test_resolve_attachment_path_empty_string_returns_none():
    from backend.builtin_mcps.macos_mail._helpers import resolve_attachment_path

    assert resolve_attachment_path("") is None


# ── Delimited record parsing --------------------------------------------------


def test_parse_records_round_trips_multiple_records():
    from backend.builtin_mcps.macos_mail._helpers import (
        FIELD_SEP, RECORD_SEP, parse_records,
    )

    fields = ["id", "subject", "sender"]
    text = (
        f"1{FIELD_SEP}Hello{FIELD_SEP}alice@example.com{RECORD_SEP}"
        f"2{FIELD_SEP}World{FIELD_SEP}bob@example.com{RECORD_SEP}"
    )
    records = parse_records(text, fields)
    assert records == [
        {"id": "1", "subject": "Hello", "sender": "alice@example.com"},
        {"id": "2", "subject": "World", "sender": "bob@example.com"},
    ]


def test_parse_records_handles_empty_field_between_separators():
    from backend.builtin_mcps.macos_mail._helpers import (
        FIELD_SEP, RECORD_SEP, parse_records,
    )

    fields = ["id", "subject", "sender"]
    text = f"1{FIELD_SEP}{FIELD_SEP}alice@example.com{RECORD_SEP}"
    records = parse_records(text, fields)
    assert records == [{"id": "1", "subject": "", "sender": "alice@example.com"}]


def test_parse_records_strips_trailing_newline_from_osascript_output():
    from backend.builtin_mcps.macos_mail._helpers import (
        FIELD_SEP, RECORD_SEP, parse_records,
    )

    fields = ["id", "subject"]
    text = f"1{FIELD_SEP}Hello{RECORD_SEP}\n"
    records = parse_records(text, fields)
    assert records == [{"id": "1", "subject": "Hello"}]


def test_parse_records_empty_text_returns_empty_list():
    from backend.builtin_mcps.macos_mail._helpers import parse_records

    assert parse_records("", ["id"]) == []
    assert parse_records("\n", ["id"]) == []


def test_parse_single_record_shape():
    from backend.builtin_mcps.macos_mail._helpers import FIELD_SEP, parse_single_record

    fields = ["subject", "sender", "content"]
    text = f"Hello{FIELD_SEP}alice@example.com{FIELD_SEP}Body text\n"
    record = parse_single_record(text, fields)
    assert record == {
        "subject": "Hello", "sender": "alice@example.com", "content": "Body text",
    }


def test_parse_single_record_empty_text_returns_blank_fields():
    from backend.builtin_mcps.macos_mail._helpers import parse_single_record

    assert parse_single_record("", ["a", "b"]) == {"a": "", "b": ""}


def test_parse_attachments_round_trips_multiple_entries():
    from backend.builtin_mcps.macos_mail._helpers import (
        ATTACH_FIELD_SEP, ATTACH_SEP, parse_attachments,
    )

    text = (
        f"report.pdf{ATTACH_FIELD_SEP}102400{ATTACH_FIELD_SEP}true"
        f"{ATTACH_SEP}"
        f"image.png{ATTACH_FIELD_SEP}2048{ATTACH_FIELD_SEP}false"
    )
    entries = parse_attachments(text)
    assert entries == [
        {"name": "report.pdf", "file_size": "102400", "downloaded": "true"},
        {"name": "image.png", "file_size": "2048", "downloaded": "false"},
    ]


def test_parse_attachments_empty_text_returns_empty_list():
    from backend.builtin_mcps.macos_mail._helpers import parse_attachments

    assert parse_attachments("") == []

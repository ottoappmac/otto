from backend.output_analysis import detect_test_command, extract_errors, summarize_tests


def test_pytest_failure_is_summarised():
    out = (
        "FAILED tests/test_a.py::test_one - assert 1 == 2\n"
        "ERROR tests/test_b.py::test_two\n"
        "=========== 1 failed, 3 passed, 1 error in 0.42s ===========\n"
    )
    report = summarize_tests(out, 1)
    assert report["status"] == "failed"
    assert "1 failed, 3 passed" in report["summary"]
    assert report["failures"] == ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]


def test_jest_and_go_and_cargo():
    jest = "  ● adds numbers\nTests:       1 failed, 2 passed, 3 total\n"
    assert summarize_tests(jest, 1)["failures"] == ["adds numbers"]
    go = "--- FAIL: TestSum (0.00s)\nFAIL\nok  \tpkg/other\t0.1s\n"
    assert summarize_tests(go, 1)["failures"] == ["TestSum"]
    cargo = "test tests::bad ... FAILED\ntest result: FAILED. 1 passed; 1 failed\n"
    assert summarize_tests(cargo, 101)["failures"] == ["tests::bad"]


def test_unknown_runner_falls_back_to_exit_code():
    assert summarize_tests("whatever", 0) == {"status": "passed", "summary": "exit code 0", "failures": []}
    assert summarize_tests("whatever", 2)["summary"] == "exit code 2"


def test_extract_errors_keeps_python_traceback_whole():
    log = (
        "starting\n"
        "Traceback (most recent call last):\n"
        '  File "app.py", line 3, in <module>\n'
        "    import missing\n"
        "ModuleNotFoundError: No module named 'missing'\n"
        "after\n"
    )
    blocks = extract_errors(log)
    assert any("ModuleNotFoundError" in block and "app.py" in block for block in blocks)


def test_extract_errors_empty_for_clean_log():
    assert extract_errors("Serving on http://localhost:3000\nready\n") == []


def test_detect_test_command(tmp_path):
    assert detect_test_command(tmp_path) == ""
    (tmp_path / "go.mod").write_text("module x\n")
    assert detect_test_command(tmp_path) == "go test ./..."
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
    assert detect_test_command(tmp_path) == "npm test --silent"


def test_detect_dev_command(tmp_path):
    from backend.output_analysis import detect_dev_command

    assert detect_dev_command(tmp_path) == ""
    (tmp_path / "Makefile").write_text("serve:\n\techo hi\n")
    assert detect_dev_command(tmp_path) == "make serve"
    (tmp_path / "package.json").write_text('{"scripts": {"start": "node .", "dev": "vite"}}')
    assert detect_dev_command(tmp_path) == "npm run dev"

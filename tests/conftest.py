"""Pytest configuration and fixtures."""

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(root_dir, "src")
tests_path = os.path.join(root_dir, "tests")
sys.path.insert(0, src_path)
sys.path.insert(0, tests_path)

os.environ["COHERE_API_KEY"] = "test-api-key"
os.environ["ENVIRONMENT_TYPE"] = "test"
os.environ["LOG_LEVEL"] = "DEBUG"
# The local API token is on by default. Tests talk to the app in-process
# and do not send it. Read at request time, so this must be set before
# the first request, which is why it lives here rather than in a fixture.
os.environ["OTTO_API_TOKEN_REQUIRED"] = "0"

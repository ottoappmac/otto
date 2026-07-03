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

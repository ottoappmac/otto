"""Post-create tests to verify the environment is correctly set up."""

import sys


def test_imports():
    """Test that essential imports work."""
    print("Testing essential imports...")

    try:
        import fastapi
        print(f"  fastapi: {fastapi.__version__}")
    except ImportError as e:
        print(f"  fastapi: FAILED - {e}")
        return False

    try:
        import langchain
        print(f"  langchain: {langchain.__version__}")
    except ImportError as e:
        print(f"  langchain: FAILED - {e}")
        return False

    try:
        import langchain_cohere  # noqa: F401
        print("  langchain_cohere: OK")
    except ImportError as e:
        print(f"  langchain_cohere: FAILED - {e}")
        return False

    try:
        import chromadb
        print(f"  chromadb: {chromadb.__version__}")
    except ImportError as e:
        print(f"  chromadb: FAILED - {e}")
        return False

    print("All essential imports successful!")
    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

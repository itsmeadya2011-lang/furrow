"""A simple hello world script."""

from __future__ import annotations


def hello() -> str:
    """Return the hello world greeting."""
    return "Hello, World!"


def main() -> None:
    """Print the hello world greeting."""
    print(hello())


if __name__ == "__main__":
    main()

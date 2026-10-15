import pytest

from hello_world import hello, main


def test_hello_returns_greeting() -> None:
    assert hello() == "Hello, World!"


def test_main_prints_greeting(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"

from click.testing import CliRunner

from furrow.cli.main import main


def test_help_succeeds():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0


def test_version_outputs_010():
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_start_help_shows_usage():
    runner = CliRunner()
    result = runner.invoke(main, ["start", "--help"])
    assert result.exit_code == 0
    assert "GOAL" in result.output

from click.testing import CliRunner

from furrow.cli.main import main


class TestCLI:
    def test_start_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["start", "--help"])
        assert result.exit_code == 0
        assert "--max-cycles" in result.output
        assert "--max-parallel-tasks" in result.output
        assert "--planner-model" in result.output
        assert "--worker-model" in result.output
        assert "--tester-model" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from furrow.cli.main import main


class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_serve_command_exists(self, runner):
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Alias for the web command" in result.output

    def test_serve_command_alias_for_web(self, runner):
        with patch("furrow.cli.main.run") as mock_run:
            result = runner.invoke(main, ["serve"])
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_web_command_exists(self, runner):
        result = runner.invoke(main, ["web", "--help"])
        assert result.exit_code == 0
        assert "Run the Furrow web server" in result.output

    def test_start_command_exists(self, runner):
        result = runner.invoke(main, ["start", "--help"])
        assert result.exit_code == 0
        assert "Run Furrow to autonomously work toward a goal" in result.output

    def test_start_handles_keyboard_interrupt(self, runner):
        with patch("furrow.cli.main.Orchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.run.side_effect = KeyboardInterrupt()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            result = runner.invoke(main, ["start", "test goal"])
            
            assert result.exit_code == 0
            assert "Furrow stopped by user" in result.output

    def test_start_handles_exception(self, runner):
        with patch("furrow.cli.main.Orchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.run.side_effect = Exception("Test error")
            mock_orchestrator_class.return_value = mock_orchestrator
            
            result = runner.invoke(main, ["start", "test goal"])
            
            assert result.exit_code == 1
            assert "Error:" in result.output
            assert "Test error" in result.output

    def test_start_with_model_override(self, runner):
        with patch("furrow.cli.main.settings") as mock_settings:
            with patch("furrow.cli.main.Orchestrator") as mock_orchestrator_class:
                mock_orchestrator = MagicMock()
                mock_orchestrator.run = MagicMock()
                mock_orchestrator_class.return_value = mock_orchestrator
                
                result = runner.invoke(main, ["start", "test goal", "--model", "custom-model"])
                
                assert result.exit_code == 0
                assert mock_settings.model == "custom-model"

    def test_start_prompts_for_goal_when_not_provided(self, runner):
        with patch("furrow.cli.main.settings") as mock_settings:
            with patch("furrow.cli.main.Orchestrator") as mock_orchestrator_class:
                with patch("furrow.cli.main.click.prompt", return_value="prompted goal") as mock_prompt:
                    mock_orchestrator = MagicMock()
                    mock_orchestrator.run = MagicMock()
                    mock_orchestrator_class.return_value = mock_orchestrator
                    
                    result = runner.invoke(main, ["start"])
                    
                    assert result.exit_code == 0
                    mock_prompt.assert_called_once_with("Enter your goal for Furrow")
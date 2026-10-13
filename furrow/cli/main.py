from __future__ import annotations

import click
from rich.console import Console

from furrow.cli.commands.start import start_cmd
from furrow.cli.commands.web import web_cmd
from furrow.logging_config import configure_logging

console = Console()


@click.group()
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
@click.option("--log-json", is_flag=True, help="Output logs as JSON")
def main(log_level: str, log_json: bool) -> None:
    configure_logging(level=log_level, json_output=log_json)


main.add_command(start_cmd)
main.add_command(web_cmd)


if __name__ == "__main__":
    main()
"""HTTP entrypoint for Furrow: the API server and its ``run`` helper.

Example:
    ```python
    from furrow.web import app, run

    run(host="127.0.0.1", port=8000)
    ```
"""

from furrow.web.server import app, run

__all__ = ["app", "run"]

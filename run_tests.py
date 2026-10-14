import subprocess
import sys

# Install the package in development mode
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
    check=False, capture_output=True,
    text=True,
    cwd="/workspace/app",
)
print(result.stdout)
print(result.stderr, file=sys.stderr)

# Run tests
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v"],
    check=False, capture_output=True,
    text=True,
    cwd="/workspace/app",
)
print(result.stdout)
print(result.stderr, file=sys.stderr)

import subprocess
import sys

# Try running pytest
result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], capture_output=True, text=True, timeout=120)
print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)
print("Return code:", result.returncode)
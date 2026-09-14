import subprocess
import sys

# Install the package in development mode
result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                       cwd="/workspace/64b5f2dc-1bcb-407e-b26a-0216ac2acba8/sessions/agent_40bb51bc-cc02-498e-bc22-22c3bc858e2c",
                       capture_output=True, text=True, timeout=120)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
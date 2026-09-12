#!/usr/bin/env python
"""Run tests and collect results."""
import subprocess
import sys
import json

def run_tests():
    # Run pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd="/workspace/64b5f2dc-1bcb-407e-b26a-0216ac2acba8/sessions/agent_d45ee109-9d1d-42e1-a4b9-86cbbdf97343"
    )
    
    # Run ruff
    ruff_result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "furrow/", "tests/"],
        capture_output=True,
        text=True,
        cwd="/workspace/64b5f2dc-1bcb-407e-b26a-0216ac2acba8/sessions/agent_d45ee109-9d1d-42e1-a4b9-86cbbdf97343"
    )
    
    # Run mypy
    mypy_result = subprocess.run(
        [sys.executable, "-m", "mypy", "furrow/", "tests/"],
        capture_output=True,
        text=True,
        cwd="/workspace/64b5f2dc-1bcb-407e-b26a-0216ac2acba8/sessions/agent_d45ee109-9d1d-42e1-a4b9-86cbbdf97343"
    )
    
    return {
        "pytest": {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        },
        "ruff": {
            "returncode": ruff_result.returncode,
            "stdout": ruff_result.stdout,
            "stderr": ruff_result.stderr
        },
        "mypy": {
            "returncode": mypy_result.returncode,
            "stdout": mypy_result.stdout,
            "stderr": mypy_result.stderr
        }
    }

if __name__ == "__main__":
    results = run_tests()
    print(json.dumps(results, indent=2))
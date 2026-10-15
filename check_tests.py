import asyncio
from furrow.agents.tester import TesterAgent
from furrow.config import settings

async def main():
    agent = TesterAgent(settings=settings)
    test_output = await agent._run_tests()
    print("Test output:")
    print(test_output)
    # Determine if tests passed
    # Typical pytest output: "...." or "F" etc.
    # We'll simple check if "FAILED" in output or "failed"
    if "FAILED" in test_output or "failed" in test_output.lower():
        passed = False
    else:
        passed = True
    print(f"Passed: {passed}")

if __name__ == "__main__":
    asyncio.run(main())
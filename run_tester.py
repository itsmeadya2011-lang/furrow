import asyncio
from furrow.agents.tester import TesterAgent
from furrow.config import settings, TaskModel

async def main():
    # Create tasks that were presumably created
    task1 = TaskModel(
        id="1",
        description="Create circle_area.py with incorrect area function (using diameter instead of radius)",
        files=["circle_area.py"],
        dependencies=[]
    )
    task2 = TaskModel(
        id="2",
        description="Create test_circle_area.py with test expecting correct area result",
        files=["test_circle_area.py"],
        dependencies=["circle_area.py"]
    )
    tasks = [task1, task2]
    
    agent = TesterAgent(settings=settings)
    goal = "Create a Python script that defines a function to calculate the area of a circle, but deliberately implement it incorrectly (use diameter instead of radius in the formula) and create a test that expects the correct result"
    result = await agent.run(goal, tasks)
    print(f"Passed: {result.passed}")
    print(f"Summary: {result.summary}")
    print(f"Failures: {result.failures}")

if __name__ == "__main__":
    asyncio.run(main())
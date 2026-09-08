# tests/test_agent_core.py
import asyncio
from services.agent.core import AegisAgent

async def test_agent():
    print("--- Testing Aegis Agent Loop with Tracing ---")
    agent = AegisAgent()
    
    # Enable event streaming to inspect tool execution
    async for event in agent.agent_executor.astream_events(
        {"messages": [("user", "Who is Roboute Guilliman and what is his role?")]},
        version="v2"
    ):
        kind = event["event"]
        if kind == "on_tool_start":
            print(f"\n[TOOL CALL] Executing {event['name']} with input: {event['data'].get('input')}")
        elif kind == "on_tool_end":
            print(f"[TOOL RESULT] {str(event['data'].get('output'))[:200]}...\n")
        elif kind == "on_chain_end" and event["name"] == "LangGraph":
            print("\n--- Final Answer ---")
            print(event["data"]["output"]["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(test_agent())
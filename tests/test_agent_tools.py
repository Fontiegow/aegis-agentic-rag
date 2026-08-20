# tests/test_agent_tools.py
from services.agent.tools import search_knowledge_base

def test_tool_execution():
    print("--- Testing Agent Tool: search_knowledge_base ---")
    tool_output = search_knowledge_base(query="Who is Roboute Guilliman?", limit=2)
    print("Tool Execution Result:\n")
    print(tool_output)

if __name__ == "__main__":
    test_tool_execution()
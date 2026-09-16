import os
from features.ui.session_state import get_agent_dependencies
from features.orchestrator.agent import TravelAgent

def test_run():
    print("Initializing agent dependencies...")
    retriever, mcp_client, context_manager, llm = get_agent_dependencies()
    agent = TravelAgent(retriever, mcp_client, context_manager, llm)
    
    query = "What is the weather like in Singapore today?"
    print(f"User Query: {query}")
    
    response = agent.process_query(query)
    print(f"\nAgent Response:")
    print(response.answer)
    print(f"\nIntent: {response.intent}")
    print(f"Tools Used: {response.mcp_tools_used}")
    print(f"KB Sources: {response.kb_sources_used}")

if __name__ == "__main__":
    test_run()

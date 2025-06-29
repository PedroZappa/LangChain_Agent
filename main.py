from rich import print
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Create the agent
def main():
    # Create the agent
    memory = MemorySaver()
    model = init_chat_model("qwen3", model_provider="ollama")
    search = TavilySearch(max_results=2)
    # Once we have all the tools we want, we can put them in a list that we will reference later.
    tools = [search]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}

    model_with_tools = model.bind_tools(tools)
    query = "Search for the weather in Povoa de Varzim, Portugal."
    response = model_with_tools.invoke(
        [{"role": "user", "content": query}],
        config=config
    )
    response.text()
    print(f"Message content: {response.text()}\n")
    print(f"Tool calls: {response.tool_calls}")

if __name__ == "__main__":
    main()

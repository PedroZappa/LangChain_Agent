from langchain.chat_models.base import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from rich import print
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


def main():
    # Create the agent
    memory = MemorySaver()
    model = init_chat_model("qwen3", model_provider="ollama")
    search = TavilySearch(max_results=2)
    tools = [search]
    agent_executor = create_react_agent(
        model, tools, checkpointer=memory
    )

    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}

    stream_agent_response(agent_executor, "Hi, I'm Zedro", config)
    stream_agent_response(agent_executor, "What's my name?", config)


def stream_agent_response(agent_executor, message_content, config) -> None:
    """
    Helper function to stream agent responses and print them.

    :param agent_executor: The agent executor.
    :param message_content: The message content.
    :param config: The config.
    """
    input_message = {"role": "user", "content": message_content}

    for step, metadata in agent_executor.stream(
        {"messages": [input_message]}, stream_mode="messages", config=config
    ):
        if metadata["langgraph_node"] == "agent" and (text := step.text()):
            print(text, end="")


if __name__ == "__main__":
    main()

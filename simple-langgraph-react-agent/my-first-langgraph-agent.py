from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json
import argparse
import os

#Check if all required environment variables are set
keys_to_check = ["LANGSMITH_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY"]
missing_keys = [key for key in keys_to_check if not os.getenv(key)]
if missing_keys:
        print(f"Exiting: Missing required environment variables -> {', '.join(missing_keys)}")
        sys.exit(0) 

print("All required environment variables are set. Proceeding...")

#Parse the command line arguments
parser = argparse.ArgumentParser(description="Weather Agent")# add an optional flag (boolean)
parser.add_argument("--prompt", type=str, default="What is the weather in San Francisco?",
                    help="Add city for the weather agent")

args = parser.parse_args()

print("User prompt:", args.prompt)

def pretty_print_agent_output(response):
    for msg in response["messages"]:
        # If it's a Pydantic model (LangChain messages are), use .dict()
        try:
            msg_dict = msg.dict()
        except AttributeError:
            # Fallback: convert generic Python objects
            msg_dict = vars(msg)

        # Pretty print as JSON
        print(json.dumps(msg_dict, indent=2, default=str))

web_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

agent = create_react_agent(
    model="gpt-4o-2024-05-13",
    tools=[web_search_tool],
    prompt="You are a helpful weather assistant. You can use the web_search tool to search the web for information.",
)

def web_search(query: str) -> str:
    return web_search_tool.run(query)

def get_final_ai_output(response):
    for msg in reversed(response["messages"]):
        # Only return the last AIMessage that has content
        if msg.__class__.__name__ == "AIMessage" and msg.content.strip():
            return msg.content
    return None

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": args.prompt}]}
)

# Example usage
output = get_final_ai_output(response)

print("\nWeather Agent Output:")
print(output)

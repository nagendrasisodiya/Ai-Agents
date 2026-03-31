import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ibm import ChatWatsonx

load_dotenv(dotenv_path=r"D:\WrokSpace\GEN-AI\AI-Agents\.env")

llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url="https://eu-de.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    api_key=os.getenv("WATSONX_API_KEY"),
)


@tool
def add(a: int, b: int) -> int:
    """
        add a and b.

        Args:
            a (int): first integer to be added
            b (int): second integer to be added

        Return:
            int: sum of a and b
    """
    return a + b

@tool
def subtract(a: int, b:int) -> int:
    """
        subtract a and b.

        Args:
            a (int): first integer to be subtracted
            b (int): second integer to be subtracted

        Return:
            int: subtraction of a and b
    """
    return a - b

@tool
def multiply(a: int, b:int) -> int:
    """
        multiply a and b.

        Args:
            a (int): first integer to be multiplied
            b (int): second integer to be multiplied

        Return:
            int: multiplication of a and b
    """
    return a * b

tool_map = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply
}

tools=[add, subtract, multiply]


# bind_tools only makes tools available. It doesn’t automatically execute them. we need either:
# A manual loop or Use LangChain’s agent framework (AgentExecutor) which handles that loop for us automatically.
llm_with_tools=llm.bind_tools(tools)


test_query = "What is 3 + 2?"
chat_history = [HumanMessage(content=test_query)]

response=llm_with_tools.invoke(chat_history)
chat_history.append(response)
print(type(response))


tool_calls = response.tool_calls
tool_1_name = tool_calls[0]["name"]
tool_1_args = tool_calls[0]["args"]
tool_call_1_id = tool_calls[0]["id"]

print(f'tool name:\n{tool_1_name}')
print(f'tool args:\n{tool_1_args}')
print(f'tool call ID:\n{tool_call_1_id}')

tool_response=tool_map[tool_1_name].invoke(tool_1_args)
print(tool_response)


# encapsulating entire manual tool call process from llm query to tool call
class ToolCallingAgent:
    def __init__(self, llm):
        self.llm_with_tools = llm.bind_tools(tools)
        self.tool_map = tool_map

    def run(self, query: str) -> str:
        chat_history = [HumanMessage(content=query)]
        response = self.llm_with_tools.invoke(chat_history)

        if not response.tool_calls:
            return response.contet

        # parsing tool call from llm
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        # manually calling tool
        tool_result = self.tool_map[tool_name].invoke(tool_args)

        tool_message = ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
        chat_history.extend([response, tool_message])

        final_response = self.llm_with_tools.invoke(chat_history)
        return final_response.content

my_agent = ToolCallingAgent(llm)
print(my_agent.run("eight multiply 2"))
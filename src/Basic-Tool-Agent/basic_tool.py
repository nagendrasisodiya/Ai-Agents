import os

from dotenv import load_dotenv
from langchain_core.tools import Tool, tool
from langchain_ibm import ChatWatsonx

load_dotenv(dotenv_path=r'D:\WrokSpace\GEN-AI\AI-Agents\.env')

llm=ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
)


# In AI, a tool will call a basic function or capability that can be called on to perform a specific task.
# Think of it like a single item in a toolbox: just like a hammer, screwdriver, or wrench, an AI toolbox is full of specific functions designed to solve problems or get things done.
#
# When building tools for tool calling, there are a few key principles to keep in mind:
#
# Clear purpose: Make sure the tool has a well-defined job.
# Standardized input: The tool should accept input in a predictable, structured format so it’s easy to use.
# Consistent output: Always return results in a format that’s easy to process or integrate with other systems.
# Comprehensive documentation: Your tool should include clear, simple documentation that explains what it does, how to use it, and any quirks or limitations.


def add_numbers(input:str)->dict:
    """
    adds a list of number provided in input string or extracts a list of number and then adds them

    :parameter
    - input(str):
    string, it should contain numbers that can be extracted and summed
    :returns
    -dict: a dictionary with a single key "result" containing the sum of numbers
    """
    numbers=[int(x) for x in input.replace(",", "").split() if x.isdigit()]
    result=sum(numbers)
    return {"result":result}


# The Tool class in LangChain serves as a structured wrapper that converts regular Python functions into agent-compatible tools.
# Each tool needs three key components:
#
# A name that identifies the tool
# The function that performs the actual operation
# A description that helps the agent understand when to use the tool

add_tool=Tool(
    name="AddTool",
    func=add_numbers,
    description="adds a list of numbers and returns the result"
)

test_input="i have 5 rupe some one gave me 10 more rupee and i found 2 rupees on road how much money i have?"
response=add_tool.invoke(test_input)
print(response) #so till now no llm is involved it just simple tool creation and invoking it

# now we had created a tool using Tool class
# we can also create a tool using just @tool decorator on the func that we want to make tool and its preferred way
# after defining a function, we can decorate it with @tool to create a tool that implements the Tool Interface.
# The @tool decorator creates a StructuredTool with schema information extracted from function signatures and docstrings

@tool
def sum_numbers(input:str)->dict:
    """
    adds a list of number provided in input string or extracts a list of number and then adds them

    :parameter
    - input(str):
    string, it should contain numbers that can be extracted and summed
    :returns
    -dict: a dictionary with a single key "result" containing the sum of numbers
    """
    numbers=[int(x) for x in input.replace(",", "").split() if x.isdigit()]
    result=sum(numbers)
    return {"result":result}

print(sum_numbers.name)
print(sum_numbers.description)
print(sum_numbers.invoke("2 3 5 s g f 5"))
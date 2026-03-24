import os
import re
from typing import Union, Dict

from dotenv import load_dotenv
from langchain_classic.agents import initialize_agent
from langchain_core.tools import Tool, tool
from langchain_ibm import ChatWatsonx

load_dotenv(dotenv_path=r'D:\WrokSpace\GEN-AI\AI-Agents\.env')

llm=ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
)

def sum_numbers_from_text(inputs:str)->dict:
    """
    adds a list of number provided in input string or extracts a list of number and then adds them

    :parameter
    - inputs(str):
    string, it should contain numbers that can be extracted and summed
    :returns
    -dict: a dictionary with a single key "result" containing the sum of numbers
    """

    numbers=[int(x) for x in inputs.replace(",", "").split() if x.isdigit()]
    result=sum(numbers)
    return {"result":result}


add_tool=Tool(
    name="AddTool",
    func=sum_numbers_from_text,
    description="adds a list of numbers and returns the result"
)

# test_input="i have 5 rupe someone gave me 10 more rupee, and i found 2 rupees on road how much money I have?"
# response=add_tool.invoke(test_input)
# print(response) #so till now no llm is involved it just simple tool creation and invoking it


# here we are connecting tools and an LLM to work together seamlessly.
# The agent uses the LLM to understand what needs to be done and decides which tool to use based on the task.
# Here's a simple overview of the key parts:
#
# Relationship between agent and LLM
# The agent acts as the decision-maker, figuring out which tools to use and when.
# The LLM is the reasoning engine. It:
# Interprets the user's input.
# Helps the agent make decisions.
# Generates a response based on the output of the tools.
# Think of the agent as the manager assigning tasks and the LLM as the brain solving problems or delegating work.


# Key parameters of initialize_agent
# tools: array of tools to be used by agent
#
# llm:the llm model that agent going to use for reason
#
# agent:
    # Specifies the reasoning framework for the agent.
    # "zero-shot-react-description" enables:
        # Zero-shot reasoning: The agent can solve tasks it hasn't seen before by thinking through the problem step by step.
        # React framework: A logical loop of:
            # Reason → Think about the task.
            # Act → Use a tool to perform an action.
            # Observe → Check the tool's output.
            # Plan → Decide what to do next.
#
# verbose:
# If True, it prints detailed logs of the agent’s thought process.
# Useful for debugging or understanding how the agent makes decisions.

agent=initialize_agent(
    [add_tool],
    llm,
    agent="zero-shot-react-description", # zero-shot-react-description expects tools to take and return plain strings
    verbose=True,
    handle_parsing_errors=True
)

response =agent.run("In 2023, the US GDP was approximately $27.72 trillion, while Canada's was around $2.14 trillion and Mexico's was about $1.79 trillion what is the total.")
print(response)


# @tool
# def sum_numbers_with_complex_output(inputs:str)->Dict[str, Union[float, str]]:
#     """
#         extracts and sum all integers and decimal numbers from the input string
#
#         :parameter
#         -input (str): a string that may contain numeric values
#         :returns
#         -dict : a dictionary with the key "result". if numbers are found in the input string, then the value is their sum(float)
#                 if no numbers are found or any error occurs, the value is the corresponding message(str)
#     """
#     matches=re.findall(r'-?\d+(?:\.\d+)?', inputs)
#     if not matches:
#         return {"result":"No number found in input"}
#     else:
#         try:
#             numbers=[float(x) for x in matches]
#             result=sum(numbers)
#             return {"result":result}
#         except Exception as e:
#             return {"result":f"error occurred during adding numbers: {e}"}
#
#


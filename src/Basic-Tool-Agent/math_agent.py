import os

from dotenv import load_dotenv
from langchain_classic.agents import initialize_agent, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ibm import ChatWatsonx

load_dotenv(dotenv_path=r'D:\WrokSpace\GEN-AI\AI-Agents\.env')

llm=ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
)

@tool
def add_numbers(inputs:str)->dict:
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


@tool
def subtract_numbers(inputs:str)->dict:
    """
    Extracts numbers from a string and performs subtraction: first number - second number - third number...
    :parameter
    - inputs(str): string containing numbers to subtract
    :returns
    - dict: a dictionary with key "result" containing the subtraction result
    """
    numbers = [int(x) for x in inputs.replace(",", "").split() if x.isdigit()]

    if not numbers:
        return {"result": 0}

    result = numbers[0]
    for num in numbers[1:]:
        result -= num
    return {"result": result}

@tool
def multiply_numbers(inputs:str)->dict:
    """
    Extracts numbers from a string and performs multiply: first number * second number * third number...

    :parameter
    - inputs(str): string containing numbers to subtract
    :returns
    - dict: a dictionary with key "result" containing the subtraction result
    """
    numbers = [int(x) for x in inputs.replace(",", "").split() if x.isdigit()]

    if not numbers:
        return {"result": 1}

    result = 1
    for num in numbers:
        result *= num
    return {"result": result}

@tool
def divide_numbers(inputs:str)->dict:
    """
    Extracts numbers from a string and calculates the result of dividing the first number by the subsequent numbers in the sequence.
       :parameter
       - inputs(str): string containing numbers to subtract
       :returns
       - dict: a dictionary with key "result" containing the subtraction result
       """
    numbers = [int(x) for x in inputs.replace(",", "").split() if x.isdigit()]

    if not numbers:
        return {"result": 1}

    result = numbers[0]
    for num in numbers[1:]:
        result /= num
    return {"result": result}

tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]

prompt = PromptTemplate(
    input_variables=["input", "tools", "agent_scratchpad", "tool_names"],
    template="""You are a helpful mathematical assistant. 
    Use the tools strictly in the following format:
    
    Question: {input}
    
    Thought: Explain your reasoning briefly.
    Action: one of [{tool_names}]
    Action Input: the input string for the tool
    
    Observation: result from the tool
    Thought: reflect on the observation
    Final Answer: the final answer to the question
    
    Tools available:
    {tools}
    
    {agent_scratchpad}"""
)


agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=prompt
)

# Wrap the agent with AgentExecutor to handle the agentic loop
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({
    "input": "What is 25 divided by 4?"
})

# Get the final answer
final_answer = response["output"]
print(final_answer)
import os

from dotenv import load_dotenv
from langchain_classic.agents import initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper
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
def search_Wikipedia(query:str)->str:
    """
    Search Wikipedia for factual information about a topic.
    Parameters:
    - query (str): The topic or question to search for on Wikipedia

    Returns:
    - str: A summary of relevant information from Wikipedia
    """
    # its a langchain builtin wikipedia wrapper
    wiki=WikipediaAPIWrapper()
    return wiki.run(query)

print(search_Wikipedia.invoke("What is tool calling?"))

agent=initialize_agent(
    [search_Wikipedia],
    llm,
    agent="zero-shot-react-description",
    handle_parsing_errors=True
)

response =agent.invoke("what is population of USA")
print(response)
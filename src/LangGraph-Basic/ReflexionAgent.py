import json
import os
from typing import List

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ibm import ChatWatsonx
from langgraph.constants import END
from langgraph.graph import MessageGraph
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=r"D:\WrokSpace\GEN-AI\AI-Agents\.env")


tavily_tool=TavilySearchAPIRetriever()
query="healthy food suggestion"
search_results=tavily_tool.invoke(query)
# print(search_results)


model_id="ibm/granite-4-h-small"
project_id= os.getenv("WATSONX_PROJECT_ID")
llm=ChatWatsonx(
    model_id=model_id,
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY"),
    project_id=project_id
)


prompt_template=ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Dr. Paul Saladino, "Carnivore MD," advocating for animal-based nutrition and challenging plant-centric dietary dogma. Focus on the superior bioavailability of animal foods and the potential toxicity of plant compounds such as oxalate, lectins, and phytates.

        Your response must follow these steps:
        1. {first_instruction}
        2. Present the evolutionary and biochemical rationale for animal-based nutrition, emphasizing nutrient density, bioavailability, and the absence of plant antinutrients.
        3. Challenge conventional "plants are healthy" narratives with mechanistic evidence about plant defense compounds and their impact on human physiology.
        4. Reflect and critique your answer. Be rigorous about evolutionary consistency and nutrient science.
        5. After the reflection, **list 1-3 search queries separately** for researching antinutrients, bioavailability studies, or evolutionary nutrition. Do not include them inside the reflection.

        Focus on: organ meats as superfoods, the carnivore elimination protocol, and why "fiber is unnecessary" based on human evolutionary biology.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        "Answer the user's question above using the required format, emphasizing the superiority of animal-based nutrition."
    ),
])

# defining responder
first_responder_prompt=prompt_template.partial(first_instruction="provide a detailed 250 word answer")
temp_chain=first_responder_prompt |llm
response=temp_chain.invoke({"messages":[HumanMessage(content=query)]})
# print(response.content)


# structuring the agent's op using data models,
# To make the agent's self-critique process reliable, we need to enforce a specific output structure.
# We use Pydantic BaseModel to define two data classes:
#
# Reflection: This class structures the self-critique, requiring the agent to identify what information is missing and what is superfluous (unnecessary).
# AnswerQuestion: This class structures the entire response.
# It forces the agent to provide its main answer, a reflection (using the Reflection class), and a list of search_queries.


class Reflection(BaseModel):
    missing:str=Field(description="what information is missing")
    superfluous:str=Field(description="what information is unnecessary")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="Main response to the question")
    reflection: Reflection = Field(description="Self-critique of the answer")
    search_queries: List[str] = Field(description="Queries for additional research")


# Now, we bind the AnswerQuestion data model as a tool to our LLM chain.
# This crucial step forces the LLM to generate its output in the exact JSON format defined by our Pydantic classes.
# The LLM doesn't just write text; it calls this "tool" to structure its entire thought process.

initial_chain=first_responder_prompt | llm.bind_tools(tools=[AnswerQuestion])
response=initial_chain.invoke({"messages":[HumanMessage(content=query)]})
# print("toolcall",response.tool_calls)


# tool execution
response_list=[]
response_list.append(HumanMessage(content=query))
response_list.append(response)

tavily_tool=TavilySearchResults(max_results=3)

def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message = state[-1]
    tool_messages = []
    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])
            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result
            tool_messages.append(ToolMessage(
                content=json.dumps(query_results),
                tool_call_id=call_id)
            )
    return tool_messages

tool_response=execute_tools(response_list)
response_list.extend(tool_response)


# defining revisor
# The Revisor is the final piece of the Reflection loop. Its job is to take the original answer, the self-critique, and the new information from the tool search, and then generate an improved, more evidence-based response.
#
# We create a new set of instructions (revise_instructions) that guide the Revisor. These instructions emphasize:
#
# Incorporating the critique.
# Adding numerical citations from the research.
# Distinguishing between correlation and causation.
# Adding a "References" section.


revise_instructions = """Revise your previous answer using the new information, applying the rigor and evidence-based approach of Dr. David Attia.
- Incorporate the previous critique to add clinically relevant information, focusing on mechanistic understanding and individual variability.
- You MUST include numerical citations referencing peer-reviewed research, randomized controlled trials, or meta-analyses to ensure medical accuracy.
- Distinguish between correlation and causation, and acknowledge limitations in current research.
- Address potential biomarker considerations (lipid panels, inflammatory markers, and so on) when relevant.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit) in the form of:
- [1] https://example.com
- [2] https://example.com
- Use the previous critique to remove speculation and ensure claims are supported by high-quality evidence. Keep response under 250 words with precision over volume.
- When discussing nutritional interventions, consider metabolic flexibility, insulin sensitivity, and individual response variability.
"""
revisor_prompt = prompt_template.partial(first_instruction=revise_instructions)

# structring revisor op
class ReviseAnswer(AnswerQuestion):
    reference:List[str]= Field(description="Citations motivating your updated answer.")

revisor_chain=revisor_prompt | llm.bind_tools(tools=[ReviseAnswer])

# invoking revisor
response=revisor_chain.invoke({"messages":response_list})
# print(response.tool_calls)


# building graph
MAX_ITERATION=4
def event_loop(state:List[BaseMessage])->str:
    count_tool_visits=sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits >= MAX_ITERATION:
        return END
    return "execute_tools"
graph=MessageGraph()

graph.add_node("respond", initial_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)

graph.add_edge("respond", "execute_tools")
graph.add_edge("execute_tools", "revisor")

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("respond")

app=graph.compile()
response=app.invoke( """I'm pre-diabetic and need to lower my blood sugar, and I have heart issues.
    What breakfast foods should I eat and avoid"""
                     )
print(response)


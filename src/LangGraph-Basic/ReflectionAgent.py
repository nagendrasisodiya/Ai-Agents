import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ibm import ChatWatsonx
from typing import List, Annotated, TypedDict, Sequence

from langgraph.constants import END
from langgraph.graph import MessageGraph


load_dotenv(dotenv_path=r"D:\WrokSpace\GEN-AI\AI-Agents\.env")


model_id="ibm/granite-4-h-small"
project_id=os.getenv("WATSONX_PROJECT_ID")
llm=ChatWatsonx(
    model_id=model_id,
    url="https://eu-de.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    api_key=os.getenv("WATSONX_API_KEY")
)



# creating generation prompt
generation_prompt=ChatPromptTemplate.from_messages([
    (
        "system",  "You are a professional LinkedIn content assistant tasked with crafting engaging, insightful, and well-structured LinkedIn posts."
            " Generate the best LinkedIn post possible for the user's request."
            " If the user provides feedback or critique, respond with a refined version of your previous attempts, improving clarity, tone, or engagement as needed.",
    ),
    MessagesPlaceholder(variable_name="messages")
])

# creating chain for post-generation
generation_chain=generation_prompt | llm



# creating reflection prompt
reflection_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional LinkedIn content strategist and thought leadership expert. Your task is to critically evaluate the given LinkedIn post and provide a comprehensive critique. Follow these guidelines:
        
        1. Assess the post’s overall quality, professionalism, and alignment with LinkedIn best practices.
        2. Evaluate the structure, tone, clarity, and readability of the post.
        3. Analyze the post’s potential for engagement (likes, comments, shares) and its effectiveness in building professional credibility.
        4. Consider the post’s relevance to the author’s industry, audience, or current trends.
        5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
        6. Evaluate the effectiveness of any call-to-action or takeaway.
        
        Provide a detailed critique that includes:
        - A brief explanation of the post’s strengths and weaknesses.
        - Specific areas that could be improved.
        - Actionable suggestions for enhancing clarity, engagement, and professionalism.
        
        Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful, constructive, and practical.
        """
    ),
    MessagesPlaceholder(variable_name="messages")
])

# creating chain for reflection
reflection_chain=reflection_prompt | llm


# defining state for agent
class AgentState(TypedDict):
    messages:Annotated[List[HumanMessage | AIMessage | SystemMessage], "add_messages"]
    # add_messages: Ensures new messages are appended to the list, preserving the context needed for iterative interactions



# LangGraph's MessageGraph
# Instead of manually defining and managing the state, LangGraph offers a prebuilt solution called MessageGraph.
# It abstracts the complexity of state management, making it easy to create and work with conversational workflows.
#
# Features of MessageGraph:
#
# Predefined State Management: Handles the underlying state representation for you, similar to what was manually defined above.
# Ease of Integration: Provides a seamless way to interact with LLMs by managing conversation states automatically.
# Workflow Simplification: Streamlines the process of building workflows with less boilerplate code.


# initializing messagegraph
graph=MessageGraph()



# Input: The function accepts the state, which is a sequence of BaseMessage objects (i.e., HumanMessage, AIMessage, SystemMessage).
# These messages provide the context necessary for generating a meaningful response.
#
# Output: The function uses the generate_chain to produce an output by invoking the chain with the state as input.
# The invoke function triggers the execution of the chain, where the messages in the state guide the chain's generation process.
# The output is generated based on the context provided by these messages, ensuring that the response is appropriate to the current stage of the conversation or task.
def generation_node(state:Sequence[BaseMessage])->List[BaseMessage]:
    generated_post=generation_chain.invoke({"messages":state})
    return [AIMessage(content=generated_post.content)]


# Input: The function takes messages, which is a sequence of BaseMessage objects.
# This includes previous AI responses, user inputs, and system-level instructions.
# The messages are used to provide context to the reflection process, guiding the generation of a more refined output.
#
# Output: The function invokes reflect_chain, passing the messages as input to critique and improve the content.
# After receiving the refined output, it returns the result as a HumanMessage object.
def reflection_agent(state:Sequence[BaseMessage])->List[BaseMessage]:
    res=reflection_chain.invoke({"messages":state})
    return [HumanMessage(content=res.content)]

# Why HumanMessage?
# The output is wrapped in a HumanMessage because the reflection process is a form of feedback or critique given to the generation agent, and the feedback is intended to be treated as if it is coming from the user.
# This is important for the iterative process where the AI generates content and then receives human-like feedback to improve the output.
# In the context of this workflow, we treat the feedback as if a human is guiding the reflection agent to enhance its output.
#
# HumanMessage here is not used to represent user input directly but rather to provide feedback (as if from a human perspective).
# This feedback is passed back into the system, enabling the generation agent to revise its content.
# It effectively gives the reflection node the authority to "speak" to the generation node, but in the context of providing critique and recommendations for refinement.


# adding nodes to graph
graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_agent)

# creating edge
graph.add_edge("reflect", "generate")

# setting entry point
graph.set_entry_point("generate")


# adding a router node for decision-making
def should_continue(state:List[BaseMessage]):
    print(state)
    print(len(state))
    if len(state)>6:
        return END
    else:
        return "reflect"

graph.add_conditional_edges("generate", should_continue)

work_flow=graph.compile()

inputs = HumanMessage(content="""Write a linkedin post on getting a software developer job at BlackRock under 160 characters""")
response=work_flow.invoke(inputs)
print(response)


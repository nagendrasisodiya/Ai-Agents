import os
from typing import TypedDict, Optional

from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from langchain_ibm import WatsonxLLM, ChatWatsonx
from langgraph.constants import END
from langgraph.graph import StateGraph

load_dotenv(dotenv_path=r"D:\WrokSpace\GEN-AI\AI-Agents\.env")

credentials=Credentials(
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY")
)

client=APIClient(
    credentials=credentials
)
# print(client.foundation_models.TextModels.show())

llm=ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url="https://eu-de.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    api_key=os.getenv("WATSONX_API_KEY"),
)


class QAState(TypedDict):
    question:Optional[str]
    context:Optional[str]
    answer:Optional[str]

# qa_state_example = QAState(
#     question="What is the purpose of this guided project?",
#     context="This project focuses on building a chatbot using Python.",
#     answer=None
# )

def input_validation_node(state):
    question=state.get("question", "").strip()
    if not question:
        return {"valid": False, "error":"question cant be empty"}
    return {"valid":True}


# This node checks if the question is related to the guided project.
# If it mentions "LangGraph" or "guided project," it provides the relevant context.
# Otherwise, it sets the context to `None
def context_provider_node(state):
    question=state.get("question", "").lower()
    if "langgraph" in question or "guided project" in question:
        context = (
            "This guided project is about using LangGraph, a Python library to design state-based workflows. "
            "LangGraph simplifies building complex applications by connecting modular nodes with conditional edges."
        )
        return {"context": context}
    return {"context":None}

def llm_node(state):
    question=state.get("question", "")
    context=state.get("context", "")

    if not context:
        return {"answer":"dont have enough context to answer question"}
    prompt=f"Context: {context}\nQuestion: {question}\nAnswer the question based on provided context"
    try:
        response=llm.invoke(prompt)
        return {"answer": response.content.strip()}
    except Exception as e:
        return {"answer": f"an error occurred during response generation: {str(e)}"}

workFlow=StateGraph(QAState)

workFlow.add_node("InputNode", input_validation_node)
workFlow.add_node("ContextNode", context_provider_node)
workFlow.add_node("QANode", llm_node)

workFlow.set_entry_point("InputNode")
workFlow.add_edge("InputNode", "ContextNode")
workFlow.add_edge("ContextNode", "QANode")
workFlow.add_edge("QANode", END)


qa_app=workFlow.compile()
result=qa_app.invoke({"question":"what day is today"})
print(result)
result=qa_app.invoke({"question": "What is LangGraph?"})
print(result)


# 1. invoke() is called with initial state
#                     ↓
# 2. Workflow starts at the entry point (first node)
#                     ↓
# 3. Node executes and returns a dictionary
#                     ↓
# 4. LangGraph AUTOMATICALLY MERGES that dictionary into the state
#                     ↓
# 5. Updated state is passed to the NEXT node
#                     ↓
# 6. Repeat steps 3-5 until the workflow reaches END
#                     ↓
# 7. invoke() RETURNS the final, accumulated state
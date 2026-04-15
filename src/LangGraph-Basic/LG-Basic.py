from typing import TypedDict, Optional
from langgraph.constants import END
from langgraph.graph import StateGraph


# states
class AuthState(TypedDict):
    # fields can be string or none
    username:Optional[str]
    password:Optional[str]
    isAuthenticated:Optional[bool]
    output:Optional[str]


# # successful state
# auth_state1:AuthState={
#     "username": "alice123",
#     "password": "123",
#     "isAuthenticated": True,
#     "output": "Login successful."
# }
# print(f"state1: {auth_state1}")
#
#
# # unsuccessful state
# auth_state2:AuthState={
#     "username": "",
#     "password": "wrongpass",
#     "isAuthenticated": True,
#     "output": "auth failed, try again latter."
# }
# print(f"state2: {auth_state2}")
#
# auth_state3:AuthState={
#     "username": "user1",
#     "password": "1234@",
#     "isAuthenticated": False,
#     "output": "auth failed, try again latter."
# }
# print(f"state2: {auth_state2}")



def input_node(state):
    print(f"state in input_node: {state}")
    username=""
    # id username doesnt exist return default ""
    if state.get("username", "")=="":
        username=input("whats your username?")
    password=input("enter your pass...")
    if state.get("username", "")=="":
        return {"username":username, "password":password}
    else:
        return {"password":password}

# print(input_node(auth_state1))
# print(input_node(auth_state2))

def validate_credentials_node(state):
    username=state.get("username", "")
    password=state.get("password", "")
    isAuthenticated=False
    print("username: ", username, "password:", password)

    if username=="user1" and password=="1234@":
        isAuthenticated=True
    return {"isAuthenticated":isAuthenticated}

# print(validate_credentials_node(auth_state1))
# print(validate_credentials_node(auth_state3))

def success_node(state):
    return {"output": "authentication successful!, welcome"}

def failure_node(state):
    return {"output": "not successfully, please try again!"}


# router node: decision-making point

def router(state):
    if state['isAuthenticated']:
        return "success_node"
    else:
        return "failure_node"


# now creating graph
# To begin building the workflow, we need to create a graph that will serve as the foundation for connecting nodes and defining the application's logic.
# We create a new instance of StateGraph using our AuthState structure, which acts as a blueprint for the application's state.
# This graph will manage the flow of execution between nodes, ensuring a seamless and organized workflow.


workflow=StateGraph(AuthState)

# adding nodes to graph using add_node method
workflow.add_node("InputNode", input_node)
workflow.add_node("ValidateCredentials", validate_credentials_node)
workflow.add_node("Success", success_node)
workflow.add_node("Failure", failure_node)

# adding edges or creating a connection between nodes using add_edge

# add_edge(start, end): This method creates a directed edge between two nodes, defining the flow of execution from one node to another.
# start: The node from which the flow begins. In this case, it's "InputNode", where the user provides their credentials.
# end: The node to which the flow leads. Here, it's "ValidateCredential", where the credentials entered by the user are validated.

workflow.add_edge("InputNode", "ValidateCredentials")
workflow.add_edge("Success", END)
workflow.add_edge("Failure", "InputNode")


# adding conditional-edge
# Validate Credentials Node: After validating the user credentials, the system uses a conditional edge to decide:
# If is_authenticated is True, the flow moves to the Success Node.
# If is_authenticated is False, the flow loops back to the InputNode so the user can try entering their credentials again

workflow.add_conditional_edges("ValidateCredentials", router, {"success_node":"Success", "failure_node":"Failure"})
# add_conditional_edges(start, router, conditions): This method defines the conditional transitions from a given node.
# start: The node where the conditional edges start (in this case, "ValidateCredential").
# router: A function that determines the condition. It checks the current state (like the is_authenticated status) and returns the appropriate node to transition to (either "Success" or "Failure").
# conditions: A dictionary that maps conditions (such as "success_node" or "failure_node") to target nodes, indicating where to direct the flow based on the condition.



# setting entry point
# set_entry_point(node): This method sets the starting point for the workflow.
# node: The name of the node where the workflow will begin.
# In this case, "InputNode", which ensures the agent prompts the user for their credentials before proceeding with the authentication process.
workflow.set_entry_point("InputNode")



# compiling the workflow
app=workflow.compile()

input_state={"username":""}
result=app.invoke(input_state)
print(result)
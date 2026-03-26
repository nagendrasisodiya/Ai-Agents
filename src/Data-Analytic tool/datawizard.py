import glob
import json
import os
from typing import Optional, List, Dict, Any, Union

import pandas as pd
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ibm import WatsonxLLM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split




load_dotenv(dotenv_path=r'D:\WrokSpace\GEN-AI\AI-Agents\.env')

credentials=Credentials(
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY")
)
client=APIClient(credentials=credentials)
print(client.foundation_models.TextModels.show())

model_id="ibm/granite-3-8b-instruct"

# adding due to model issue
# the actual value passed to your tool becomes '["classification_dataset.csv", "regression-dataset.csv"]\nObservation', which is invalid.
# This is a model formatting problem — Granite isn't stopping after Action Input
def clean_action_input(raw: str) -> str:
    """Strip anything after a newline that looks like agent scaffolding."""
    for marker in ["\nObservation", "\nThought", "\nAction", "\nFinal"]:
        if marker in raw:
            raw = raw[:raw.index(marker)]
    return raw.strip()



@tool
def list_csv_files()->Optional[List[str]]:
    """
    lists all csv files in the current directory
    :return:
        a list containing all csv file names.
        and if no csv files are found, returns None.
    """
    csv_files=glob.glob(os.path.join(os.getcwd(), "*.csv"))
    # os.getcwd() retrieves the current working directory.
    # os.path.join(os.getcwd(), "*.csv") constructs a path pattern to match all CSV files (* matches all filenames ending with .csv).
    # glob.glob(pattern) returns a list of files that match the given pattern.

    if not csv_files:
        return None
    return csv_files


# dataset caching
DATAFRAME_CACHE = {}
# when using functions in Python with LangChain agents, simply writing the keyword global isn't enough to maintain data between different tool calls.
# This is because each time the agent runs a tool, it might do so in a separate execution environment, causing any global variables to reset.
# Instead, using a module-level dictionary (DATAFRAME_CACHE) that lives outside any function creates a persistent storage space that all tools can access without explicitly passing it around.
# This approach works reliably across multiple function calls, even when they happen in different contexts, and keeps the function interfaces clean by avoiding the need to pass the cache as an additional parameter to every tool.

@tool
def preload_dataset(paths:str)->str:
    """
    loads csv files into a global cache if not already loaded

        This function helps to efficiently manage datasets by loading them once
        and storing them in memory for future use. Without caching, we would
        waste tokens describing dataset contents repeatedly in agent responses.

    :param
        paths: a JSON string representing a list of file paths
    :return:
        a message summerizing which dataset was loaded or already cached
    """
    paths = clean_action_input(paths)
    try:
        path_list = json.loads(paths)
    except (json.JSONDecodeError, TypeError):
        path_list = [paths.strip()]

    loaded=[]
    cached=[]
    for path in path_list:
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path]=pd.read_csv(path)
            loaded.append(path)
        else:
            cached.append(path)
    return (
        f"Loaded dataset: {loaded}\n"
        f"already cached: {cached}"
    )

# summarizing tool
@tool
def get_dataset_summaries(dataset_paths:str)->List[Dict[str, Any]]:
    """
    analyzes multiple csv files and returns metadata summeries for each one
    :param
        dataset_paths: a JSON string representing a list of file paths
    :return:
        List[Dict[str, Any]]:
            a list of summaries, one per dataset each containing
            - "file_name": The path of the dataset file.
            - "column_names": A list of column names in the dataset.
            - "data_types": A dictionary mapping column names to their data types (as strings).
    """
    dataset_paths = clean_action_input(dataset_paths)
    try:
        path_list = json.loads(dataset_paths)
    except (json.JSONDecodeError, TypeError):
        path_list = [dataset_paths.strip()]

    summaries=[]
    for path in path_list:
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path]=pd.read_csv(path)

        df=DATAFRAME_CACHE[path]

        summary={
            "file_name": path,
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict()
        }
        summaries.append(summary)
    return summaries


@tool
def call_dataframe_method(file_name:str, method:str)->str:
    """
    Execute a method on a DataFrame and return the result.
    This tool lets you run simple DataFrame methods like 'head', 'tail', or 'describe'
    on a dataset that has already been loaded and cached using 'preload_datasets'.

    Param
    :param
        file_name(str):the path or file name of dataset in global cache
    :param
        method(str):the name of the method to call on a dataframe.Only no-argument
         methods are supported (e.g., 'head', 'describe', 'info').
    :return:
        Str:the output of a method as a formated string, or an error message if the dataset
        is not found or the method is invalid.

    Example:
       call_dataframe_method(file_name="data.csv", method="head")
    """
    file_name= clean_action_input(file_name)
    method= clean_action_input(method)
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name]=pd.read_csv(file_name)
        except FileNotFoundError as e:
            return f"dataframe {file_name} is not found in cache and disk"
        except Exception as e:
            return f"Error in file loading: {e}"

    df=DATAFRAME_CACHE[file_name]
    func=getattr(df, method, None) #getattr() function, which allows us to retrieve and call a method using its string name.
    if not callable(func):
        return f"method={method} is an invalid method on dataframe"
    try:
        result=func()
        return str(result)
    except Exception as e:
        return f"Error calling {method} on {file_name} : {e}"


@tool
def evaluate_classification_dataset(file_name:str, target_column:str)->Dict[str, Union[str, float]]:
    """
    train and evaluate a classifier on a dataset using specified targeted columns.
    Param
    :param
        file_name: name or path to dataset stored in DATAFRAME_CACHE
    :param
        target_column: The name of the column to use as the classification target.
    :return:
        Dict[str, float]: A dictionary with the model's accuracy score.
    """
    file_name = clean_action_input(file_name)
    target_column = clean_action_input(target_column)
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}

    df = DATAFRAME_CACHE[file_name]
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc}


@tool
def evaluate_regression_dataset(file_name:str, target_column:str)->Dict[str, Union[str, float]]:
    """
    Train and evaluate a regression model on a dataset using the specified target column.

    :param
        file_name: name or path to dataset stored in DATAFRAME_CACHE
    :param
        target_column: The name of the column to use as the classification target.

    :return:
        Dict[str, float]: A dictionary with the model's accuracy score.
    """
    file_name = clean_action_input(file_name)
    target_column = clean_action_input(target_column)
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}

    df = DATAFRAME_CACHE[file_name]
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "r2_score": r2,
        "mean_squared_error": mse
    }


# making a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data science assistant. Use the available tools to analyze CSV files.\n\n"
     "You have access to the following tools:\n{tools}\n\n"
     "STRICTLY follow this format and do not deviate:\n\n"
     "Question: the input question you must answer\n"
     "Thought: think about what to do next\n"
     "Action: the tool to use, must be one of [{tool_names}]\n"
     "Action Input: the input to the tool as a plain string\n"
     "Observation: the result of the tool (this is provided to you)\n"
     "... (repeat Thought/Action/Action Input/Observation as needed)\n"
     "Thought: I now know the final answer\n"
     "Final Answer: the final answer to the original question\n\n"
     "IMPORTANT: After 'Action Input:', stop and wait. Never write 'Observation:' yourself."),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

llm_model=WatsonxLLM(
    model_id="ibm/granite-4-h-small",
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params={
        "max_new_tokens": 1024,
        "temperature": 0.0,
    }
)
tools=[list_csv_files, preload_dataset,
       get_dataset_summaries,
       call_dataframe_method,
       evaluate_classification_dataset,
       evaluate_regression_dataset
       ]

agent = create_react_agent(llm_model, tools, prompt)

# Wrap it in an executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# response = agent_executor.invoke({
#     "input": "Can you tell me about the dataset?"
# })
# print(response)

print(" Ask questions about  dataset (type 'exit' to quit):")

while True:
    user_input = input(" You:")
    if user_input.strip().lower() in ['exit', 'quit']:
        print("bye byee")
        break

    result = agent_executor.invoke({"input": user_input})
    print(f"Agent: {result['output']}")
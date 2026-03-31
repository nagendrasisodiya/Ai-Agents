import os

import pandas as pd
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv(dotenv_path=r"D:\WrokSpace\GEN-AI\AI-Agents\.env")

credentials=Credentials(
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY")
)
client=APIClient(credentials=credentials)
# print(client.foundation_models.TextModels.show())

model_id="meta-llama/llama-3-3-70b-instruct"

params={
    GenParams.MAX_NEW_TOKENS:256,
    GenParams.TEMPERATURE:0
}

model=Model(
    model_id=model_id,
    credentials=credentials,
    params=params,
    project_id=os.getenv("WATSONX_PROJECT_ID")
)
llm=WatsonxLLM(model=model)


data_file=pd.read_csv("DataSets/StudentPerformanceFactors.csv")
# print(data_file.head(5))

# langchain pandas agent
agent=create_pandas_dataframe_agent(
    llm=llm,
    df=data_file,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True
)
response=agent.invoke("how many rows in this file")
print(response['output'])

response=agent.invoke("how any students have bellow 70 percent attendance and how many of them are male")
print(response['output'])
# print(response['intermediate_steps'][-1][0].tool_input.replace(":", "\n"))

response=agent.invoke("How does study time affect exam results")
print(response['output'])

response=agent.invoke("Create a bar chart showing average performance by parental education level.")
print(response['output'])
print(response['intermediate_steps'][-1][0].tool_input.replace(":", "\n"))

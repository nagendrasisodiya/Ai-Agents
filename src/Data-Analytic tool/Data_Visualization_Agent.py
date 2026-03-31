import os

import pandas as pd
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv(dotenv_path=r"D:\WrokSpace\GEN-AI\AI-Agents\.env")


data_file=pd.read_csv("DataSets/Student Alcohol Consumption DataSet/student-mat.csv")

credentials=Credentials(
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY")
)
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
llm_model=WatsonxLLM(model)


agent=create_pandas_dataframe_agent(
    llm=llm_model,
    df=data_file,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True
)

# response = agent.invoke(
#     "Generate scatter plots showing the relationship between "
#     "'Medu' (mother's education level) and 'G3' (final grade), "
#     "and between 'Fedu' (father's education level) and 'G3'. "
#     "and also tell me whats the insight you found "
#
# )
# print(response['output'])

response = agent.invoke("Use bar plots to compare the average final grades ('G3') of students with internet access at home versus those without ('internet' column).")
print(response['output'])

import os

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_classic.agents import AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

load_dotenv(dotenv_path=r"D:\WrokSpace\GEN-AI\AI-Agents\.env")

credentials=Credentials(
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY")
)
# client=APIClient(credentials=credentials)
# print(client.foundation_models.TextModels.show())

model_id="meta-llama/llama-3-3-70b-instruct"
params = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.0,
    GenParams.TOP_P: 1.0,
    GenParams.REPETITION_PENALTY: 1.05,
}
model=Model(
    model_id=model_id,
    credentials=credentials,
    params=params,
    project_id=os.getenv("WATSONX_PROJECT_ID")
)
llm_model=WatsonxLLM(model=model)


mysql_username=os.getenv("MYSQL_USERNAME")
mysql_password=os.getenv("MYSQL_PASSWORD")
mysql_host=os.getenv("MYSQL_HOST")
mysql_port=os.getenv("MYSQL_PORT")
database_name=os.getenv("DATABASE_NAME")
mysql_uri = f'mysql+mysqlconnector://{mysql_username}:{mysql_password}@{mysql_host}:{mysql_port}/{database_name}'
db = SQLDatabase.from_uri(mysql_uri)


sql_agent=create_sql_agent(
    llm=llm_model,
    db=db,
    verbose=True,
    handle_parsing_errors=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

response=sql_agent.invoke("how many albums are thre in databse")
print(response['output'])

# response=sql_agent.invoke("Can you left join table Artist and table Album by ArtistId? Please show me 5 Name and AlbumId in the joint table.")
# print(response['output'])
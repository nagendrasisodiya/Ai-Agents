# AI-Agents: Advanced LLM Orchestration Framework

A comprehensive collection of autonomous agents and tool-calling implementations leveraging Large Language Models (LLMs) for complex task automation, data analytics, and database interaction. This project demonstrates sophisticated engineering patterns in agentic workflows, primarily using the LangChain ecosystem and IBM WatsonX AI services.

---

## 🏗️ Project Architecture

The repository is structured into modular components, each focusing on a specific domain of agentic behavior:

```text
AI-Agents/
├── src/
│   ├── Basic-Tool-Agent/          # Foundation of tool-augmented LLM reasoning
│   │   ├── basic_agent.py         # ReAct framework implementation (Reason + Act)
│   │   ├── math_agent.py          # Arithmetic & logic specialized agents
│   │   └── wikipedia_tool_agent.py # Real-world information retrieval integration
│   ├── Data-Analytic tool/        # Autonomous data science & visualization
│   │   ├── DataSets/              # Curated datasets for agent training/testing
│   │   ├── Data_Visualization_Agent.py # Automated plotting & insight generation
│   │   ├── datawizard.py          # Custom DF manipulation & ML evaluation tools
│   │   └── naturalLang_to_dataVisualization.py
│   ├── LangChain SQL-Agent/       # Natural Language to Structured Query (Text-to-SQL)
│   │   └── sql_agent_basic.py     # Dynamic SQL generation & execution
│   └── Manual-Tool-Calling/       # Low-level control of tool selection logic
│       └── basic_manual_tool_calling.py
├── .env                           # Environment configuration (Keys & Endpoints)
├── pyproject.toml                 # Poetry dependency management
└── README.md                      # Project documentation
```

---

## 🚀 Key Features

### 1. Autonomous Data Analyst
Utilizes the `create_pandas_dataframe_agent` to transform raw CSV data into actionable insights.
- **Dynamic Visualization:** Generates complex Matplotlib/Seaborn plots from natural language prompts.
- **ML Evaluation:** Automated routines for classification and regression performance benchmarking via `datawizard.py`.

### 2. Intelligent SQL Orchestrator
Features a robust `create_sql_agent` capable of bridging the gap between human language and relational databases.
- **Schema Awareness:** Automatically introspects database schema to generate accurate JOINs and aggregations.
- **Safe Execution:** Implements error handling and parsing logic for reliable query execution.

### 3. ReAct Reasoning Loop
Implements the **Reasoning and Acting (ReAct)** pattern, allowing agents to:
- **Think:** Break down complex queries into logical steps.
- **Act:** Execute specific tools (Wikipedia, Custom Math Tools, etc.).
- **Observe:** Incorporate tool outputs back into the reasoning chain.

### 4. Enterprise-Grade LLM Integration
Powered by **IBM WatsonX AI**, utilizing state-of-the-art models:
- `meta-llama/llama-3-3-70b-instruct`
- `ibm/granite-3-8b-instruct`

---

## 🛠️ Technical Stack

- **Core Framework:** [LangChain](https://github.com/langchain-ai/langchain) / [LangGraph](https://github.com/langchain-ai/langgraph)
- **AI Backend:** [IBM WatsonX AI](https://www.ibm.com/watsonx)
- **Data Processing:** Pandas, NumPy, Scikit-Learn
- **Database:** MySQL (via SQLAlchemy & MySQL-Connector)
- **Environment:** Python 3.11+, Poetry for dependency resolution

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11 or 3.12
- IBM Cloud account with WatsonX AI access
- MySQL Server (for SQL agent features)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AI-Agents.git
   cd AI-Agents
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   WATSONX_API_KEY=your_api_key_here
   WATSONX_PROJECT_ID=your_project_id_here
   MYSQL_USERNAME=username
   MYSQL_PASSWORD=your_password
   MYSQL_HOST=localhost
   MYSQL_PORT=database_portNo
   DATABASE_NAME=your_db_name
   ```

---

## 📖 Usage Examples

### Running the Data Visualization Agent
```python
# src/Data-Analytic tool/Data_Visualization_Agent.py
response = agent.invoke("Generate bar plots to compare average final grades ('G3') by internet access.")
```

### Querying your Database in Natural Language
```python
# src/LangChain SQL-Agent/sql_agent_basic.py
response = sql_agent.invoke("How many albums are there in the database?")
```

---

## 🛡️ Engineering Best Practices
- **Environment Isolation:** Strict use of `.env` for credential management.
- **Tool Decoupling:** Tools are defined as standalone functions or classes, ensuring high testability.
- **Verbose Debugging:** Agents are configured with `verbose=True` to allow full transparency into the "Chain of Thought" (CoT).


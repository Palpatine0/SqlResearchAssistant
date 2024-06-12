## SQL Research Assistant

### Project Introduction
The SQL Research Assistant project is an advanced tool designed to perform research by executing SQL queries on databases. It integrates LangChain's research assistant template with SQL query capabilities, leveraging local models like Llama and GPT to generate and refine queries. This project demonstrates the integration of AI with structured data, providing comprehensive research reports and deploying the application using Lang Serve for user-friendly interaction.

### Objective
To set up a SQL Research Assistant that can generate, refine, and execute SQL queries on databases using LangChain and Lang Serve.

### Prerequisites
- Python 3.11
- pip (Python package installer)
- Git (optional)

### Step 1: Initial Setup

#### 1. Initialize the Environment
First, let's set up the environment and install necessary dependencies.


1. **Create a `.env` file:**
   - This file will store your API keys and other configuration settings. Ensure it is included in your `.gitignore` file to prevent it from being committed to your repository.

   Example `.env` file:
   ```plaintext
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_API_KEY="your_langchain_api_key"
   LANGCHAIN_PROJECT="SqlResearchAssistant"
   OPENAI_API_KEY="your_open_api_key"
   ```
   

2. **Install required packages:**
   ```bash
   pip install langchain langchain_community openai streamlit python-dotenv
   ```
   ```bash
   pip install -U langchain-cli
   ```
   ```bash
   pip install -U duckduckgo-search
   ```
   ```bash
   pip install beautifulsoup4
   ```

#### Key Concepts

##### 1. DuckDuckGo Search API
- **Definition**: DuckDuckGo Search API is a tool that allows developers to access DuckDuckGo search results programmatically. It provides an easy way to integrate web search functionality into applications.
- **Usage**: It is used in this project to perform web searches and retrieve links to relevant web pages based on a user's query.

##### 2. BeautifulSoup

- **Definition**: BeautifulSoup is a Python library used for parsing HTML and XML documents. It creates a parse tree for parsing HTML and XML documents to extract data from HTML, which is useful for web scraping.
- **Usage**: BeautifulSoup is typically used in conjunction with requests to fetch and parse web pages. It allows you to navigate the parse tree and search for specific elements, such as tags, attributes, and text.


### Step 2: Setup LangServe and LangSmith

#### 1. LangServe Setup
Set up LangServe to manage our application deployment.

1. **Initialize a New LangServe Application:**
   - Use the LangServe CLI to create a new application called `sql-research-assistant`.

   Command:
   ```bash
   langchain app new sql-research-assistant
   ```
#### 2. LangSmith Setup

Make sure u have created a LangSmith project for this lab.

**Project Name:** SqlResearchAssistant


### Step 3: feat: Integrate research assistant modules and add web scraping
Copied `search`, `chain.py`, and `writer.py` from LangChain's research assistant template.

#### 1. Copy directory `search` under the directory `sql-research-assistant/app`

**Copy the Search Directory:**
Copy the `search` directory from the research-assistant template [here](https://github.com/langchain-ai/langchain/tree/master/templates/research-assistant/research_assistant/search) 
to the `sql-research-assistant/app` directory.
 


#### 2. Update the Chain Module

**File**: `sql-research-assistant/app/chain.py`

**Changes**:
- Updated import paths to reflect the correct module.

<img src="https://i.imghippo.com/files/gyPeH1717755285.jpg" alt="" border="0">


**Explanation of Each File:**

- **`chain.py`**:
  - **Purpose**: This file defines the main chain of operations for the SQL Research Assistant.
  - **Main Function**: Combines different chains (search chain and writer chain) and assigns types to the input.


- **`search/web.py`**:
  - **Purpose**: Implements web search functionality using different APIs.
  - **Main Function**: Contains functions for scraping web pages, performing web searches, and defining prompts for generating search queries.
    - **Main Chain:** `chain` combines all the steps: generating search queries, performing the search, scraping the content, summarizing the results, and formatting the final response.

  The web.py file is designed to handle web search, scraping, and summarization tasks using the LangChain framework. 
  This setup allows users to perform comprehensive web searches, scrape relevant information, and generate detailed summaries, all through a structured and automated workflow using LangChain.

- **`writer.py`**:
  - **Purpose**: Implements the writer chain that generates detailed reports.
  - **Main Function**: Defines templates for various types of reports and configures the model to generate outputs based on these templates.



### Step 3: Adding SQL Query Generation and Answering Chains

In this step, we will integrate SQL query generation and answering chains into the SQL Research Assistant project.


#### 1. Add SQLite Database

**File**: `sql-research-assistant/app/search/nba_roster.db`

- Added `nba_roster.db` to the search directory to provide a sample SQLite database for testing SQL query generation and answering.

The db file can be found in [here](https://github.com/langchain-ai/langchain/tree/master/templates/sql-llama2/sql_llama2)

#### 2. Create SQL Query Generation and Answering Chains

**File**: `sql-research-assistant/app/search/sql.py`

- Created `sql.py` with SQL query generation and answering chains using LangChain and ChatOllama.

```python
# sql-research-assistant/app/search/sql.py
from pathlib import Path

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

openAI_llm = "gpt-3.5-turbo"
llm = ChatOpenAI(model=openAI_llm)


db_path = Path(__file__).parent / "nba_roster.db"
rel = db_path.relative_to(Path.cwd())
db_string = f"sqlite:///{rel}"
db = SQLDatabase.from_uri(db_string, sample_rows_in_table_info=5)


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)


# Prompt
template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""  # noqa: E501
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
        ("human", template),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

# Chain to query with memory

sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
    )
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
    | (lambda x: x.split("\n\n")[0])
)

# Chain to answer
template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""  # noqa: E501
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural "
            "language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)


# Supply the input types to the prompt
class InputType(BaseModel):
    question: str


sql_answer_chain = (
    RunnablePassthrough.assign(query=sql_chain).with_types(
        input_type=InputType
    )
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    ) | RunnablePassthrough.assign(
        answer=prompt_response
        | llm
        | StrOutputParser()
    ) | (lambda x: f"Questions: {x['question']}\n\nAnswer: {x['answer']}")

)

```

The `sql.py` file is designed to handle SQL query generation and answering using the LangChain framework and OpenAI. It achieves this through several components:

This setup allows users to ask questions in natural language.
Have those questions converted into SQL queries, execute those queries on a database, and then receive the results back in natural language. 
This integration showcases the power of combining language models with structured data for research and data analysis tasks.

#### 3. Update the `web.py` File

**File**: `sql-research-assistant/app/search/sql.py`
``- Updated `web.py` to include the SQL answer chain in the main search and response chain.``

<img src="https://i.imghippo.com/files/xrONu1717756537.png" alt="" border="0">
<img src="https://i.imghippo.com/files/v33xc1717756556.png" alt="" border="0">

**Explanation**:
The `web.py` file has been updated to include the `sql_answer_chain` in the main search and response chain. This change ensures that SQL queries generated from the user's questions are executed on the database, and the responses are then converted into natural language answers. The integration of `sql_answer_chain` into the main chain allows for a seamless process where user questions are answered using a combination of web search results and database queries.


#### Key Concepts

##### 1. SQL Database
- **Definition**: A SQL database is a structured collection of data that is stored and accessed electronically. SQL (Structured Query Language) is used to manage and manipulate the data.
- **Usage**: In this project, the `nba_roster.db` is used as a sample SQLite database to test SQL query generation and answering.
- **Example**:
  ```python
  from langchain_community.utilities import SQLDatabase
  db = SQLDatabase.from_uri("sqlite:///nba_roster.db")
  ```

##### 2. Chat Prompt Template
- **Definition**: A Chat Prompt Template defines the structure and content of prompts used to interact with language models.
- **Usage**: Templates are used to convert input questions into SQL queries and convert SQL responses into natural language answers.
- **Example**:
  ```python
  from langchain_core.prompts import ChatPromptTemplate
  template = """Based on the table schema below, write a SQL query that would answer the user's question:
  {schema}
  
  Question: {question}
  SQL Query:"""
  prompt = ChatPromptTemplate.from_messages(
      [
          ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
          ("human", template),
      ]
  )
  ```
  
##### 3. RunnablePassthrough
- **Definition**: `RunnablePassthrough is a component in the LangChain framework that allows data to pass through unchanged or be modified as needed within a chain of operations.
- **Usage**:  In this project, `RunnablePassthrough` is used to pass data between different stages of the chain and to assign the output of one stage as the input to the next.

### Step 4: Enhancing Environment Management and Tracing

In this step, we will enhance the environment management and tracing capabilities of the SQL Research Assistant project.

#### 1. Update the Chain Module

**File**: `sql-research-assistant/app/chain.py`

**Changes**:
- Add the dotenv for environment variable management, and the tracer setup to ensure tracing completion.
- Included a main function to test chain invocation with a sample question.

```python
from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

load_dotenv()
wait_for_all_tracers()
```

```python
if __name__ == "__main__":
    input_data = {
        "question": "Who is older? Point guards or Centers?"
    }

    print(chain.invoke(input_data))

```
<img src="https://i.imghippo.com/files/AnZKb1717756986.jpg" alt="" border="0">

#### 2. Update the Web Module

**File**: `sql-research-assistant/app/search/web.py`

**Changes**:
- Added dotenv for environment variable management.
- Added tracer setup to ensure tracing completion.
- Integrated `sql_answer_chain` into the main search and response chain.

<img src="https://i.imghippo.com/files/wbhrR1717701550.jpg" alt="" border="0">
<img src="https://i.imghippo.com/files/hvoFi1717701606.jpg" alt="" border="0">



#### 3. Run the Main Chain

Run `chain.py` then check the LangSmith status

#### 4. Inspect the model running process from LangSmith
Go to your [LangSmith dashboard](https://smith.langchain.com/) and check the running process of your model. 

<img src="https://i.imghippo.com/files/XmY9A1717736372.jpg" alt="" border="0">
<img src="https://i.imghippo.com/files/KsZIO1717736473.jpg" alt="" border="0">
<img src="https://i.imghippo.com/files/trNRT1717736814.jpg" alt="" border="0">

### Step 5: Serve the Application Using LangServe

#### 1. Update `server.py`:
   - Integrate the chain with FastAPI.
   - Add a route to serve the chain.

#### 2. Update `chian.py` and `web.py`:
   - Alter the import route to serve the chain.

**chain.py:**
```python
from app.search.web import chain as search_chain
from app.writer import chain as writer_chain
```

**web.py:**
```python
from app.search.sql import sql_answer_chain
```

<img src="https://i.imghippo.com/files/9W8uc1717759364.jpg" alt="" border="0">

#### 3. Serving the Application by LangServe

Run the following commands to set up and serve the application using LangServe.

   ```bash
   cd sql-research-assistant
   langchain serve
   ```

You can now access the application through the following links:

Access [Playground](http://127.0.0.1:8000/sql-research-assistant/playground/)

<img src="https://i.imghippo.com/files/tpufz1717757599.jpg" alt="" border="0">
<img src="https://i.imghippo.com/files/uabCN1717757920.jpg" alt="" border="0">
<img src="https://i.imghippo.com/files/oA0VL1717757992.jpg" alt="" border="0">
<img src="https://i.imghippo.com/files/jaGUZ1717758024.jpg" alt="" border="0">

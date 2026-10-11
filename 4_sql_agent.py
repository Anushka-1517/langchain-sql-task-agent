from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st


# Connect to the SQLite database
db = SQLDatabase.from_uri("sqlite:///my_tasks.db")


# Create the tasks table if it does not exist
db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK (
            status IN ('pending', 'in_progress', 'completed')
        ) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")


print("Database connected")
print("Available tables:", db.get_usable_table_names())


# Initialize the Groq model
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# Create SQL tools
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()


# Instructions for the agent
system_prompt = """
You are a task management assistant that interacts with a SQL database
containing a 'tasks' table.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC.
2. After CREATE, UPDATE, or DELETE, confirm the operation with a SELECT query.
3. If the user requests a list of tasks, present the output in a structured table.
4. Never invent task records or claim an operation succeeded unless verified.
5. Use only the tasks table for task management.

CRUD OPERATIONS:
CREATE: INSERT INTO tasks(title, description, status)
READ: SELECT * FROM tasks WHERE ... LIMIT 10
UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema:
id, title, description, status (pending/in_progress/completed), created_at.
"""


# Create the agent once and preserve its memory
@st.cache_resource
def get_agent():
    memory = InMemorySaver()

    agent_instance = create_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        system_prompt=system_prompt
    )

    return agent_instance


agent = get_agent()


# Streamlit UI
st.title("TaskBot")
st.caption("AI-powered task management using LangChain, Groq, and SQLite")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


# Get user input
prompt = st.chat_input("Ask me to manage your tasks...")


if prompt:
    # Display and save the user's message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    st.chat_message("user").markdown(prompt)

    # Generate the agent's response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                response = agent.invoke(
                    {
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": "1"
                        }
                    }
                )

                result = response["messages"][-1].content

                st.markdown(result)

                # Save the assistant's response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")